import torch

from math import log2
from utils import TensorImageUtils, partial_generate
from ProGAN import ProGenerator

"""
Visualize: Generate a serial of fade-in images
"""

folder="tmp/save_13_52_44/"
num_alphas=10
num_alphas += 1
fixed_batch_size = 64
nrow=8

max_resolution = 1024
max_num_blocks = 6 - 2
depths = [-1] + [i for i in range(max_num_blocks)] # predefined depths
dynamic_batch_size = [32, 32, 32, 32, 24, 16, 6, 3, 2] # predefined batch size to avoid excceeding gpu memory

netG = ProGenerator(resolution=max_resolution)
netG.cuda()

utiler = TensorImageUtils("tmp/test", preprocess_func=
        lambda x:  torch.nn.functional.interpolate(x, size=(64,64))
        )

fixed_noise = torch.randn((fixed_batch_size, 512), dtype=torch.float32).cuda()

for i, depth in enumerate(depths):

    bs = dynamic_batch_size[i]

    res = int(pow(2, depth + 3))
    netG.load_state_dict(torch.load(folder + "netG{}x{}.pt".format(res, res)))

    if num_alphas == 0:
        alphas = [1.0]
    else:
        alphas = torch.linspace(0, 1, steps=num_alphas)
    for alpha in alphas:
        image = partial_generate(netG, fixed_noise, bs, depth=depth, alpha=alpha)
        utiler.save_images(image, "test%dx%d_%.2f.png" %(res, res, alpha), nrow=nrow)

    print("Finished: {}".format(res))
