"""
Generate images with differenct resolution.
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

z = torch.randn((batch_size, dim_z), dtype=torch.float32).cuda()

images = netG(z, depth, 1.0)

utiler.save_images(images, "inference.png", nrow=8)
print("Inference Complete")
