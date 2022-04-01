import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, variant='A'):
        super(ConvAE, self).__init__()

        if variant == 'A':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 12, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(12, 6, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 3, 3, padding=1)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(3, 6, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(6, 12, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(12, 1, 5, padding=2)
            )
        elif variant == 'B':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 24, 5, padding=2),
                nn.LeakyReLU(0.05, inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 12, 3, padding=1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(12, 6, 3, padding=1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.MaxPool2d(2)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(6, 6, 3, padding=1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(6, 12, 3, padding=1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(12, 24, 3, padding=1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(24, 1, 5, padding=2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 24, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(24, 12, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(12, 6, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(6, 6, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(6, 12, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(12, 24, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.ConvTranspose2d(24, 1, 7, padding=3),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

