"""CompactCNN: the binary real-vs-generated classifier, trained from scratch.

The same architecture is used for every equalization strategy and every source generator.
Holding the model fixed is what makes the four transfer matrices comparable, so nothing
here should be tuned per-arm.

Input:  (B, 3, 128, 128) float32 in [-1, 1]   -- see normalize() below
Output: (B, 1) raw logits, positive = synthetic

Self-check:  python src/model.py
"""

import torch
import torch.nn as nn

# A mistyped width still gives a network that trains, just not the one being described.
# The smoke test asserts this value so that cannot pass unnoticed.
EXPECTED_PARAMS = 1_173_473

# Same, for the experimental two-layer head at its default width. Adding one hidden layer of
# 256 units costs 256*256 + 256 = 65,792 weights and replaces nothing, so the total is the
# baseline plus that. Asserted in the smoke test for the same reason as above.
EXPECTED_PARAMS_FC_HIDDEN_256 = EXPECTED_PARAMS + 65_792


def _block(c_in, c_out):
    """[Conv3x3 -> BN -> ReLU] x2 -> MaxPool2: c_in -> c_out channels, spatial size halved.

    padding=1 keeps the convolutions size-preserving, so the pool is the only thing in the
    network that changes resolution.
    """
    return nn.Sequential(
        # bias=False: the BatchNorm that follows has a shift parameter doing the same job.
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CompactCNN(nn.Module):
    """~1.17M-parameter binary classifier, trained from scratch.

    forward: (B, 3, 128, 128) float32 in [-1, 1] -> (B, 1) float32 logits.
    """

    def __init__(self, dropout=0.3, fc_hidden=None):
        """
        Args:
            dropout: Dropout probability, applied to the pooled 256-vector and again
                inside the two-layer head when `fc_hidden` is set.
            fc_hidden: Width of an optional hidden layer in the classifier head. `None`
                reproduces the reported architecture exactly - a single linear layer -
                and is the default so that loading any of the 56 committed checkpoints
                keeps working unchanged. An int adds Linear -> ReLU -> Dropout before
                the output layer.

                **Experimental.** The head sits after `AdaptiveAvgPool2d(1)`, so it sees
                a 256-vector with all spatial structure already averaged away; capacity
                added here cannot recover a cue the convolutions did not extract. The
                train/val diagnostic also puts this model at 0.967-0.982 train accuracy,
                i.e. variance-limited, where added capacity normally hurts. It is
                therefore paired with D4 augmentation in `train.py`, which is what buys
                the room for it - do not run it without.
        """
        super().__init__()

        # Widths 32/64/128/256; spatial 128 -> 64 -> 32 -> 16 -> 8.
        #
        # Nothing downsamples before the pool at the end of block 1. The cue separating
        # real from generated is high-frequency, so an early stride-2 stem would discard
        # it before any weight saw it. Each output cell ends up with a 76x76 receptive
        # field, 59% of the image width.
        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )

        # Average each 8x8 map instead of flattening it. The evidence here is a texture
        # statistic with no fixed location, so a head that weighted positions separately
        # would just learn the framing of the training crops.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

        # Raw logit, no sigmoid: BCEWithLogitsLoss applies it internally in a numerically
        # stable way, and attribution needs the unsquashed logit -- a sigmoid flattens the
        # gradients of confident predictions, which are the ones worth looking at.
        #
        # The two-layer form keeps that property: only the final layer is widened into,
        # and it is still a bare Linear. A second Dropout sits between the layers because
        # the hidden units are the new capacity and are the thing at risk of memorizing.
        self.fc_hidden = fc_hidden
        if fc_hidden is None:
            self.fc = nn.Linear(256, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(256, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize(x_uint8):
    """Cached images -> model input. (B, 128, 128, 3) uint8 -> (B, 3, 128, 128) float32 in [-1, 1].

    Kept next to the architecture so training and attribution cannot end up on different
    input scales, which would make their gradients incomparable without failing visibly.
    `train._normalize` is the same expression, character for character, for that reason.

    In-place after the cast: `.float()` on a uint8 source always allocates, so the ops
    that follow cannot reach back into the cached uint8 arm, and the whole mapping costs
    one allocation instead of one per operation - ~100 MB a step at the eval batch size.
    """
    return x_uint8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def _smoke_test():
    """Check the architecture without any data: output shape, parameter count, gradient flow.

    Runs in a few seconds on CPU, so a fresh clone can be verified before the cache exists.
    """
    torch.manual_seed(0)

    model = CompactCNN()
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 1), logits.shape
    assert torch.isfinite(logits).all(), "non-finite logits at init"

    n_params = count_parameters(model)
    assert n_params == EXPECTED_PARAMS, "got {}, expected {}".format(n_params, EXPECTED_PARAMS)
    print("params {:,} OK; forward {} OK".format(n_params, tuple(logits.shape)))

    # The experimental head. Checked in the same place so a change to either form cannot
    # land without the other being re-counted, and so the default stays provably identical
    # to the architecture the 56 reported runs used.
    deep = CompactCNN(fc_hidden=256)
    deep_logits = deep(torch.randn(2, 3, 128, 128))
    assert deep_logits.shape == (2, 1), deep_logits.shape
    assert torch.isfinite(deep_logits).all(), "non-finite logits at init (fc_hidden)"
    n_deep = count_parameters(deep)
    assert n_deep == EXPECTED_PARAMS_FC_HIDDEN_256, "got {}, expected {}".format(
        n_deep, EXPECTED_PARAMS_FC_HIDDEN_256
    )
    print("params {:,} OK (fc_hidden=256, +{:,})".format(n_deep, n_deep - n_params))

    # Memorize ten fixed examples. A model that cannot do this has a broken gradient path
    # somewhere (a detached tensor or a dead ReLU stack) which the shape check above
    # would pass without noticing. The labels are random, so memorizing is the only way
    # the loss can reach zero.
    model.train()
    images = torch.randn(10, 3, 128, 128)
    targets = torch.randint(0, 2, (10, 1)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01, "overfit test failed: loss {:.4f}".format(loss.item())
    print("overfit test OK (loss {:.5f})".format(loss.item()))


if __name__ == "__main__":
    _smoke_test()
