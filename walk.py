"""
Walk the linear interpolation in latent space
"""

import torch

from ProGAN import ProGenerator
from utils import TensorImageUtils
from math import log2

model_path = "tmp/save_22_41_33/netG64x64.pt"
max_resolution = 1024

resolution = 64 # resolution for output

netG = ProGenerator(resolution=max_resolution)
netG.load_state_dict(torch.load(model_path, map_location="cpu"))

use_cuda = True
if use_cuda:
    netG.cuda()

dim_z = 512
batch_size = 64

utiler = TensorImageUtils()

tores = lambda x: int(log2(x))
depth = tores(resolution) - 3

# z = torch.randn((batch_size, dim_z), dtype=torch.float32).cuda()
torch.manual_seed(1)

with torch.no_grad():

    z1 = torch.randn((1, dim_z), dtype=torch.float32).cuda()
    z2 = torch.randn((1, dim_z), dtype=torch.float32).cuda()

    alphas = torch.linspace(0, 1, steps=batch_size).view(-1, 1).repeat(1, dim_z).cuda()
    alphas = (z2 - z1) * alphas
    z = z1.repeat(batch_size, 1)
    z = z + alphas

    images = netG(z, depth, 1.0)

utiler.save_images(images, "walk.png", nrow=8)
print("Walk Complete")
