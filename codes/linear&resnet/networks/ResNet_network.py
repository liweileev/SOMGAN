import functools
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ResBlockGenerator(nn.Module):
    """ResBlockGenerator.
        (x -> BN -> ReLU -> Upsample -> conv -> BN -> ReLU -> conv) + (x -> Upsample)
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1) #(3,1,1) size don't change
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        if in_channels == out_channels:
            self.bypass = nn.Sequential(
                Upsample(scale_factor=2)
                )
        else:
            self.bypass = nn.Sequential(
                Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 1, 1, padding=0) #(1,1,0) change #channel
                )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockDiscriminator(nn.Module):
    """ResBlockDiscriminator.
        downsample: (x -> ReLU -> BN -> ReLU -> BN) + x
        no downsample: (x -> ReLU -> BN -> ReLU -> BN -> AvgPooling) + (x -> conv -> BN -> AVGPooling)
    """

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if downsample:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, padding=0)
                )
        else:# no downsample
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2,
                nn.BatchNorm2d(out_channels)
                )

        if in_channels != out_channels:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            if downsample:
                self.bypass = nn.Sequential(
                    self.bypass_conv,
                    nn.BatchNorm2d(out_channels),
                    nn.AvgPool2d(2, padding=0)
                )
            else:
                self.bypass = nn.Sequential(
                    self.bypass_conv,
                    nn.BatchNorm2d(out_channels)
                )
        else:
            if downsample:
                self.bypass = nn.Sequential(
                    nn.AvgPool2d(2, padding=0)
                )
            else:
                self.bypass = nn.Sequential()

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    """FirstResBlockDiscriminator.
        (x -> conv -> BN -> ReLU -> conv -> BN -> AvgPooling) + (x -> AvgPooling -> conv -> BN)
    """

    def __init__(self, in_channels, out_channels):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # don't apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(2) # downsample
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2), # downsample
            self.bypass_conv, # change num of channel
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    """Generator.
        Linear+reshape(=decov) -> ResBlockGenerator -> ... -> BN -> ReLU -> conv -> Tanh
    """
    
    def __init__(self, image_size=64, z_dim=100, channel=3, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.conv_dim = conv_dim

        repeat_num = math.floor(np.log2(self.imsize)) - 2
        mult = 2 ** repeat_num
        initsize = int(self.imsize / mult)

        self.l0 = nn.ConvTranspose2d(z_dim, conv_dim, initsize, 1, 0) # same as Linear
        nn.init.xavier_uniform_(self.l0.weight.data, 1.)

        self.l1 = ResBlockGenerator(conv_dim, conv_dim)

        self.l2 = ResBlockGenerator(conv_dim, conv_dim)

        if self.imsize >= 32:        
            self.l3 = ResBlockGenerator(conv_dim, conv_dim)
        
        if self.imsize >= 64:        
            self.l4 = ResBlockGenerator(conv_dim, conv_dim)
            
        if self.imsize >= 128:
            self.l5 = ResBlockGenerator(conv_dim, conv_dim)
        
        if self.imsize >= 256:
            self.l6 = ResBlockGenerator(conv_dim, conv_dim)
            
        if self.imsize >= 512:
            self.l7 = ResBlockGenerator(conv_dim, conv_dim)

        self.lastconv = nn.Conv2d(conv_dim, channel, 3, 1, 1)
        nn.init.xavier_uniform_(self.lastconv.weight.data, 1.)

        self.last = nn.Sequential(
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            self.lastconv,
            nn.Tanh()
            )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)

        out = self.l0(z) # Linear
        out = self.l1(out) # ResBlock
        out = self.l2(out) # ResBlock
        
        if self.imsize >= 32:
            out = self.l3(out)
            if self.imsize >= 64:
                out = self.l4(out)
                if self.imsize >= 128:
                    out = self.l5(out)
                    if self.imsize >= 256:
                        out = self.l6(out)
                        if self.imsize >= 512:
                            out = self.l7(out)  

        out = self.last(out)        
        return out

class Discriminator(nn.Module):
    """Discriminator.
    """

    def __init__(self, image_size=64, channel=3, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size

        self.l0 = FirstResBlockDiscriminator(channel, conv_dim) # downsample

        self.l1 = ResBlockDiscriminator(conv_dim, conv_dim, downsample=True) # downsample

        self.l2 = ResBlockDiscriminator(conv_dim, conv_dim) #size don't downsample

        if self.imsize >= 32:
            self.l3 = ResBlockDiscriminator(conv_dim, conv_dim) # size don't change

        if self.imsize >= 64:
            self.l4 = ResBlockDiscriminator(conv_dim, conv_dim)
            
        if self.imsize >= 128:
            self.l5 = ResBlockDiscriminator(conv_dim, conv_dim)
            
        if self.imsize >= 256:
            self.l6 = ResBlockDiscriminator(conv_dim, conv_dim)
            
        if self.imsize >= 512:
            self.l7 = ResBlockDiscriminator(conv_dim, conv_dim)

        self.lastconv = nn.Conv2d(conv_dim, 1, 1, 1, 0) # same as Linear
        nn.init.xavier_uniform_(self.lastconv.weight.data, 1.)

        self.last = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(int(self.imsize / 4)),
            self.lastconv
            )        

    def forward(self, x):
        out = self.l0(x)
        out = self.l1(out)
        out = self.l2(out)
        
        if self.imsize >= 32:
            out = self.l3(out)
            if self.imsize >= 64:
                out = self.l4(out)
                if self.imsize >= 128:
                    out = self.l5(out)
                    if self.imsize >= 256:
                        out = self.l6(out)
                        if self.imsize >= 512:
                            out = self.l7(out)    
        
        out = self.last(out)
        
        return out.squeeze()