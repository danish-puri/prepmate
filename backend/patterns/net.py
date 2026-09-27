"""The pretrained trunk: a small residual CNN over the 8x8 board.

Deliberately small. The job is not to play chess, it is to place positions in a
space where similar problems land near each other, and the whole pipeline is
built around running this thing as few times as possible (store.py, replay.py).
A 64-channel, 4-block trunk is about 400k parameters and encodes several
thousand positions a second on a laptop CPU, which keeps a scout interactive.

Two heads share the trunk:
  embed()  - the L2-normalized vector everything downstream clusters on
  policy() - from-square and to-square logits, the pretext task that teaches
             the trunk what matters, and later the source of the predictability
             signals in mining.py

Convolutions are the right prior here: threats, pawn chains and king shelter
are local patterns that mean the same thing wherever they sit on the board.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_PLANES


class _Block(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class PositionNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, dim: int = 128, **_):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[_Block(channels) for _ in range(blocks)])

        self.embed_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(8 * 64, dim))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Flatten())
        self.from_fc = nn.Linear(16 * 64, 64)
        self.to_fc = nn.Linear(16 * 64, 64)

    def forward(self, x):
        h = self.trunk(self.stem(x))
        p = self.policy_head(h)
        return self.from_fc(p), self.to_fc(p), F.normalize(self.embed_head(h), dim=1)

    def embed(self, x) -> torch.Tensor:
        """Unit vectors, so cosine similarity is a dot product and k-means on
        them is spherical k-means."""
        return F.normalize(self.embed_head(self.trunk(self.stem(x))), dim=1)

    def policy(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.policy_head(self.trunk(self.stem(x)))
        return F.log_softmax(self.from_fc(p), dim=1), F.log_softmax(self.to_fc(p), dim=1)
