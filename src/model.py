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

    def __init__(self, dropout=0.3):
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
        self.fc = nn.Linear(256, 1)

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
    """
    return x_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1.0


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
