import torch.nn as nn
import numpy as np
import math


class Generator(nn.Module):
    """Generator.
        deconv -> BN -> Relu -> ... deconv -> Tanh
    """

    def __init__(self, image_size=64, z_dim=100, channel=3, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        last = []

        repeat_num = math.ceil(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num
        if self.imsize == 96: #STL-10
            layer1.append(nn.ConvTranspose2d(z_dim, conv_dim * mult, 6))
        elif self.imsize == 28: #MNIST
            layer1.append(nn.ConvTranspose2d(z_dim, conv_dim * mult, 7))
        else:
            layer1.append(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4))

        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        if self.imsize >= 32:        
            layer3 = []
            layer3.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer3.append(nn.ReLU())
            self.l3 = nn.Sequential(*layer3)
            curr_dim = int(curr_dim / 2)
        
        if self.imsize >= 64:        
            layer4 = []
            layer4.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())      
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)
            
        if self.imsize >= 128:
            layer5 = []
            layer5.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer5.append(nn.ReLU())
            self.l5 = nn.Sequential(*layer5)
            curr_dim = int(curr_dim / 2)
        
        if self.imsize >= 256:
            layer6 = []
            layer6.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layer6.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer6.append(nn.ReLU())
            self.l6 = nn.Sequential(*layer6)
            curr_dim = int(curr_dim / 2)
            
        if self.imsize >= 512:
            layer7 = []
            layer7.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layer7.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer7.append(nn.ReLU())
            self.l7 = nn.Sequential(*layer7)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        
        last.append(nn.ConvTranspose2d(curr_dim, channel, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        
        if self.imsize >= 32:
            out=self.l3(out)
            if self.imsize >= 64:
                out=self.l4(out)
                if self.imsize >= 128:
                    out=self.l5(out)
                    if self.imsize >= 256:
                        out=self.l6(out)
                        if self.imsize >= 512:
                            out=self.l7(out)    

        out=self.last(out)        
        return out


class Discriminator(nn.Module):
    """Discriminator.
        conv -> BN -> LeakyReLU -> ... -> conv -> BN -> LeakyReLU
    """
    
    def __init__(self, image_size=64, channel=3, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        last = []

        if self.imsize == 96: #STL-10
            layer1.append(nn.Conv2d(channel, conv_dim, 3, 3, 0))
        elif self.imsize == 28: #MNIST
            layer1.append(nn.Conv2d(channel, conv_dim, 4, 4, 2))
        else:
            layer1.append(nn.Conv2d(channel, conv_dim, 4, 2, 1))
            
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer2.append(nn.BatchNorm2d(int(curr_dim * 2)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize >= 32:
            layer3 = []
            layer3.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer3.append(nn.BatchNorm2d(int(curr_dim * 2)))
            layer3.append(nn.LeakyReLU(0.1))
            self.l3 = nn.Sequential(*layer3)
            curr_dim = curr_dim * 2

        if self.imsize >= 64:
            layer4 = []
            layer4.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer4.append(nn.BatchNorm2d(int(curr_dim * 2)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
            
        if self.imsize >= 128:
            layer5 = []
            layer5.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer5.append(nn.BatchNorm2d(int(curr_dim * 2)))
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)
            curr_dim = curr_dim*2
            
        if self.imsize >= 256:
            layer6 = []
            layer6.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer6.append(nn.BatchNorm2d(int(curr_dim * 2)))
            layer6.append(nn.LeakyReLU(0.1))
            self.l6 = nn.Sequential(*layer6)
            curr_dim = curr_dim*2
            
        if self.imsize >= 512:
            layer7 = []
            layer7.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layer7.append(nn.BatchNorm2d(int(curr_dim * 2)))
            layer7.append(nn.LeakyReLU(0.1))
            self.l7 = nn.Sequential(*layer7)
            curr_dim = curr_dim*2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)

        last.append(nn.Conv2d(curr_dim, 1, 4))
#         last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)
        

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        
        if self.imsize >= 32:
            out = self.l3(out)
            if self.imsize >= 64:
                out=self.l4(out)
                if self.imsize >= 128:
                    out=self.l5(out)
                    if self.imsize >= 256:
                        out=self.l6(out)
                        if self.imsize >= 512:
                            out=self.l7(out)    
        
        out=self.last(out)
        
        return out.squeeze()