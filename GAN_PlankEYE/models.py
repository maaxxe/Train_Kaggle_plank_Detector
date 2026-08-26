from __future__ import annotations

import torch
import torch.nn as nn


def _down(in_ch, out_ch, normalize=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    Générateur conditionnel.
    Entrée = masque (1 canal) + bruit latent spatial.
    Sortie = RGB dans [-1, 1].

    Architecture prévue pour 512x512.
    """

    def __init__(self, latent_channels=3, base=64):
        super().__init__()
        in_ch = 1 + latent_channels

        self.d1 = _down(in_ch, base, normalize=False)       # 512 -> 256
        self.d2 = _down(base, base * 2)                    # 256 -> 128
        self.d3 = _down(base * 2, base * 4)                # 128 -> 64
        self.d4 = _down(base * 4, base * 8)                # 64 -> 32
        self.d5 = _down(base * 8, base * 8)                # 32 -> 16
        self.d6 = _down(base * 8, base * 8)                # 16 -> 8
        self.d7 = _down(base * 8, base * 8, normalize=False)  # 8 -> 4

        self.u1 = _up(base * 8, base * 8, dropout=0.5)     # 4 -> 8
        self.u2 = _up(base * 16, base * 8, dropout=0.5)    # 8 -> 16
        self.u3 = _up(base * 16, base * 8, dropout=0.5)    # 16 -> 32
        self.u4 = _up(base * 16, base * 4)                 # 32 -> 64
        self.u5 = _up(base * 8, base * 2)                  # 64 -> 128
        self.u6 = _up(base * 4, base)                      # 128 -> 256

        self.out = nn.Sequential(
            nn.ConvTranspose2d(
                base * 2, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, mask, noise):
        x = torch.cat([mask, noise], dim=1)

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        u1 = self.u1(d7)
        u1 = torch.cat([u1, d6], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d5], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d4], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d3], dim=1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, d2], dim=1)

        u6 = self.u6(u5)
        u6 = torch.cat([u6, d1], dim=1)

        return self.out(u6)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN conditionnel:
        entrée = masque + image RGB.
    """

    def __init__(self, base=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, mask, image):
        return self.net(torch.cat([mask, image], dim=1))


def init_weights(module):
    classname = module.__class__.__name__

    if "Conv" in classname and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    if "Norm" in classname and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)

    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
