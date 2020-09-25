""" Minimum Implementation of ProGAN, for default settings: + Mini-Stddev Layer + Weight scale during forward + PixelNorm after every 3x3 conv layer + Apply bias progressive method: + 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

def lerp(a, b, t): return a * t + b * (1 - t)

def change_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad(requires_grad)

class BiasApply(nn.Module):
    """ noise after some activation layer, noise added to each channel
    """
    def __init__(self, in_channels):
        super(BiasApply, self).__init__()
        self.bias = nn.Parameter(torch.zeros(in_channels, dtype=torch.float32))

    def forward(self, x):
        if len(x.size()) == 4:
            return x + self.bias.view(1, -1, 1, 1)
        else:
            return x + self.bias


class FC(nn.Module):
    """ fully-connected layer with weight scale
    """
    def __init__(self, inf, outf):
        super(FC, self).__init__()

        self.inf = inf
        self.outf = outf
        self.weights = nn.Parameter(torch.randn((outf, inf)))
        self.bias = nn.Parameter(torch.zeros(outf))
        fan_in = inf
        self.he_std = 2 ** 0.5 * (1 / fan_in) ** 0.5

    def forward(self, x):
        return F.linear(x, self.weights / self.he_std, self.bias)


class Conv2d(nn.Module):
    """ convolutional layer with weight scale, using weight scale as default
    """
    def __init__(self, inc, outc, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()

        if isinstance(kernel_size, tuple):
            h, w = kernel_size
        else:
            h, w = kernel_size, kernel_size
        self.inc = inc
        self.outc = outc
        self.h, self.w = h, w
        self.kernels = nn.Parameter(torch.randn(outc, inc, h, w))
        self.bias = nn.Parameter(torch.zeros(outc))
        fan_in = inc * h * w
        self.he_std = (2 ** 0.5) * ((1 / fan_in) ** 0.5)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.kernels * self.he_std, bias=self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return "Conv2d: [{}x{}x{}x{}]".format(self.outc, self.inc, self.h, self.w)


class PixelNorm(nn.Module):
    """PixelNorm: normalize feature vector of each pixel to unit vector
    """
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt( torch.mean( x ** 2, dim = 1, keepdim=True) + self.eps )


class MiniBatchStdDev(nn.Module):
    """MiniBatchStdDev: append at the last layer of D to improve the diversity.
    """
    def __init__(self, group_size=1000):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        gs = min(self.group_size, x.size(0))
        size = x.size()
        std = x.view(gs, -1, size[1], size[2], size[3])
        std = torch.mean(torch.std(std, dim=0), dim=[1,2,3], keepdim=True)
        std = std.repeat(gs, 1, size[2], size[3])
        return torch.cat([x,std], dim=1) 


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.down = nn.AvgPool2d(3, 2, 1)
    
    def forward(self, x):
        return self.down(x)


class ProGenerator(nn.Module):
    """ Generator of ProGAN 
    """
    def __init__(self, 
            resolution=1024, 
            fmap_base=8192, 
            fmap_decay=1.0, 
            fmap_max=512,
            apply_noise=True, # TODO: for no noise apply version.
            ):
        super(ProGenerator, self).__init__()

        self.res = int(log2(resolution))
        self.net = nn.ModuleList([])
        self.num_blocks = self.res - 2

        nf = lambda i: min(int(fmap_base/ (2 ** (i * fmap_decay))), fmap_max)

        # first block
        inc, outc = nf(0), nf(1) # 512, 512
        self.first = nn.Sequential(
            PixelNorm(), # TODO: optional for future design
            nn.ConvTranspose2d(inc, outc, 4, 1, 0), # use convtranpose2d instead of dense layer in original code for decreasing the model size.
            BiasApply(outc),
            nn.LeakyReLU(0.2, True),
            PixelNorm(),
            Conv2d(inc, outc, 3, 1, 1),
            BiasApply(outc),
            nn.LeakyReLU(0.2, True),
            PixelNorm(),
        )

        self.rgbconverters = nn.ModuleList([])
        def add_torgb(in_channels):
            self.rgbconverters.append(
                nn.Sequential(
                    Conv2d(in_channels, 3, 1, 1),
                    BiasApply(3)
                )
            )

        add_torgb(outc)

        # build remaining block iteratively
        for r in range(2, self.res):
            inc, outc = nf(r-1), nf(r)
            block = [
                # TODO: fused scale up
                Upsample(),
                Conv2d(inc, outc, 3, 1, 1),
                BiasApply(outc),
                nn.LeakyReLU(0.2, True),
                PixelNorm(),
                Conv2d(outc, outc, 3, 1, 1),
                BiasApply(outc),
                nn.LeakyReLU(0.2, True),
                PixelNorm(),
            ]

            self.net.append(nn.Sequential(*block))
            add_torgb(outc)
        self.up = Upsample()

    def forward(self, z, depth, alpha):
        """ progressive forward, same depth and alpha for the same growing structure in discriminator

        :params:
            x:      (tensor)         - latent input
            depth:  (int)            - index of blocks to perform pregressive growing ( start from 1 )
            alpha:  (float)          - fade-in parameter, ratio of direct output
        """
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.first(x)

        # depth: specify fade-in blocks in training. 0 <= depth <= num(blocks) - 1
        # depth < 0 for no fade-in blocks in training.
        if depth >= self.num_blocks: depth = self.num_blocks - 1
        if depth >= 0:
            # growing part
            for block in self.net[:depth]:
                x = block(x)

            residual = self.rgbconverters[depth](self.up(x))
            direct = self.rgbconverters[depth+1](self.net[depth](x))
            x = lerp(direct, residual, alpha)
            return x

        else: # last layer
            return self.rgbconverters[0](x)

    # TODO: freeze option
    def freeze_by_depth(self, depth):
        """ freeze unused layers in training to avoid unneccesary calculation.
        """
        pass


    # TODO: unfreeze option
    def unfreeze_by_depth(self, depth):
        pass


class ProDiscriminator(nn.Module):
    """ Discriminator of ProGAN
    """
    def __init__(self, 
            resolution=1024, 
            fmap_base=8192, 
            fmap_decay=1.0, 
            fmap_max=512):
        super(ProDiscriminator, self).__init__()

        self.res = int(log2(resolution))
        self.num_blocks = self.res - 2
        self.net = nn.ModuleList([])
        self.rgbconverters = nn.ModuleList()

        nf = lambda i: min(int(fmap_base / (2 ** (i * fmap_decay))), fmap_max )

        def add_fromrgb_layers(inc):
            self.rgbconverters.append(
                    nn.Sequential(
                        Conv2d(3, inc, 1, 1),
                        BiasApply(inc),
                        nn.LeakyReLU(0.2, True)
                    )
            )

        for r in range(self.res + 1, 3, -1):
            inc, outc = nf(r), nf(r - 1)
            block = [
                Conv2d(inc, outc, 3, 1, 1),
                BiasApply(outc),
                nn.LeakyReLU(0.2, True),
                Conv2d(outc, outc, 3, 1, 1),
                BiasApply(outc),
                nn.LeakyReLU(0.2, True),
                Downsample(),
            ]
            self.net.append(nn.Sequential(*block))
            add_fromrgb_layers(inc)

        # last layer of Discriminator
        inc, outc = nf(2), nf(1)
        self.last = nn.Sequential(
            MiniBatchStdDev(),
            Conv2d(inc + 1, outc, 3, 1, 1),
            BiasApply(outc),
            nn.LeakyReLU(0.2, True),
            Conv2d(outc, outc, 4, 1), # use Conv2d for ouput instead of dense
            BiasApply(outc),
            nn.LeakyReLU(0.2, True),
        )

        add_fromrgb_layers(inc)
        self.act = FC(outc, 1) # Default unconditional structure TODO: add conditional structure

        self.down = Downsample()

    def forward(self, x, depth, alpha):
        """ progressive forward

        :params:
            x:      (tensor)         - latent input
            depth:  (int)            - index of blocks to perform pregressive growing ( start from 0 )
            alpha:  (float)          - fade-in parameter, ratio of direct output
        """

        # depth: specify last `depth` fade-in blocks in training. 0 <= depth <= num(blocks) - 1
        # depth == num(blocks) - 1 for all blocks in training.
        # depth < 0 for no fade-in blocks in training.
        # so when training ProGAN, depth should grow up from -1 to num(blocks) - 1

        if depth >= self.num_blocks: depth = self.num_blocks - 1
        if depth >= 0:
            # growing part
            residual = self.down(self.rgbconverters[-depth-1](x))
            direct = self.net[-depth-1](self.rgbconverters[-depth-2](x))

            x = lerp(direct, residual, alpha)
            # remaining blocks
            for block in self.net[self.num_blocks - depth:]:
                for name, layer in block.named_children():
                    x = layer(x)

        else:
            x = self.rgbconverters[-1](x)

        x = self.last(x).view(x.size(0), -1)
        return self.act(x)


def test_model(depth, alpha):
    netG = ProGenerator()
    netD = ProDiscriminator()
    print(netG)
    print(netD)
    z = torch.randn((1, 512))
    x = netG(z, depth, alpha)
    print(x.size())
    y = netD(x, depth, alpha)
    print(y.size())


if __name__ == '__main__':
    test_model(8, 1)
