import time, os
import gc
gc.collect()
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

notebook = False
if notebook:
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

#------------------ configuratoin -------------------------
from opt import get_progan_options, choose_dataset
opt = get_progan_options()

from utils import test_and_add_postfix_dir, test_and_make_dir, currentTime, \
    TensorImageUtils, save_model
from data import get_mnist, get_cifar10, get_fashion

from WGANLoss import calculate_gradient_penalty
from ProGAN import ProGenerator, ProDiscriminator

#------------------ config by options ---------------------
# TODO: restructure the configuration.
resolution = 1024

# data related configuration
test = opt.test

# persistence related configuration
save_path = test_and_add_postfix_dir("tmp" + os.sep + "save_" + currentTime()) 
test_and_make_dir(save_path)
writer = SummaryWriter(save_path)
utiler = TensorImageUtils(save_path)

# training related configuration
use_cuda = opt.cuda
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
lr_D = opt.lrD
lr_G = opt.lrG
epochs = opt.epochs
save_epoch_interval = opt.save_epoch_interval

# model related parameters are excluded from opt
# define model here
dim_z = 512
netD = ProDiscriminator(resolution=resolution)
netG = ProGenerator(resolution=resolution)
if use_cuda:
    netD.cuda()
    netG.cuda()
betas = opt.adam_betas

# model parameters
opt.num_workers = 4 # recommend small num_worker

lambda_gp = 10
epsilon_drift = 0.001 # do not use drift loss currently

# trained resolution
start_resolution = 4
end_resolution = 64
first_stablize_train = True

# num_images = 8000000 # use 800000 images for training one iteration
num_images = 300000 # number of images in one training iteration
knum = num_images // 1000 # train each alpha for 1000 images

tores = lambda x: int(log2(x))
min_res = tores(4)
max_res = tores(resolution)
# start_res, end_res 控制需要训练的 blocks.
start_res = max(min_res, tores(start_resolution))
end_res = min(max_res, tores(end_resolution))

max_num_blocks = max_res - 2

depths = [-1] + [ i // 2 for i in range(max_num_blocks * 2)] # predefined depths
dynamic_batch_size = [32, 32, 32, 32, 16, 10, 6, 3, 2] # predefined batch size to avoid excceeding gpu memory

#------------------ Training -------------------------

def get_optimizer_by_depth(depth):

    if depth < 0:
        optimizer_D = optim.Adam([
            {"params": netD.last.parameters()},     
            {"params": netD.act.parameters()},              
            {"params": netD.rgbconverters[-1].parameters()}
            ], lr=lr_D, betas=betas)

        optimizer_G = optim.Adam([
            {"params": netG.first.parameters()},
            {"params": netG.rgbconverters[0].parameters()}
            ], lr=lr_G, betas=betas)
    else:
        nblocks = netD.num_blocks
        optimizer_D = optim.Adam([
            {"params": netD.last.parameters()},     
            {"params": netD.act.parameters()},              
            {"params": netD.rgbconverters[nblocks-depth-1:].parameters()}, 
            {"params": netD.net[nblocks-depth-1:].parameters()} 
            ], lr=lr_D, betas=betas)

        optimizer_G = optim.Adam([
            {"params": netG.first.parameters()},
            {"params": netG.rgbconverters[:depth+2].parameters()},
            {"params": netG.net[:depth+1].parameters()}
            ], lr=lr_G, betas=betas)

    return optimizer_D, optimizer_G

print("Start Training, using {}".format(device))
starttime = time.clock()

# original GAN Loss
criterion = nn.BCEWithLogitsLoss()
zero = torch.tensor(0, dtype=torch.float).to(device)
one = torch.tensor(1, dtype=torch.float).to(device)

# progressive training.
iter_step = 0
for i, depth in enumerate(depths):

    stablize = (i % 2 == 0)

    res = 2 ** (depth + 3)
    if res < start_resolution or res > end_resolution: continue # train in resolution range
    if res == start_resolution and not stablize: continue # start from resolution and stabliaztoin mode

    print("depth:{} - res:{}x{} - training mode:{}".format(
        depth, res, res, "stablize" if stablize else "progressive"
    ))

    # ============ redefine dataset ========================
    bs = dynamic_batch_size[depth + 1]
    opt.batch_size = bs
    opt.input_size = res
    data = choose_dataset(opt) # change batch_size and input_size
    print("Using batch_size: {}".format(bs))

    # fixed noise
    fix_noise = torch.randn((bs, dim_z), dtype=torch.float32).to(device)
    setattr(opt, "nrow", bs)

    # ============ redefine optimizer ======================
    optimizer_D, optimizer_G = get_optimizer_by_depth(depth)

    # =========== training =================================
    image_step = 0
    while image_step < num_images:
        bar = tqdm(data)
        for j, batch in enumerate(bar):
            images, _ = batch
            if use_cuda:
                images = images.cuda()

            batch_size = images.size(0)
            image_step += batch_size

            if stablize:
                alpha = 1.0
            else:
                alpha = float(min(image_step // 1000, knum) / knum) # for each alpha train for 1000 images

            # =========================
            # Update Discriminator 
            # =========================
            optimizer_D.zero_grad()

            z = torch.randn((batch_size, dim_z), device=device)
            fake = netG(z, depth, alpha)
            out_fake = netD(fake.detach(), depth, alpha)
            out_true = netD(images, depth, alpha)

            # true_label = one.expand_as(out_true)
            # fake_label = zero.expand_as(out_fake)
            
            # loss_fake = criterion(out_fake, fake_label)
            # loss_true = criterion(out_true, true_label)
            # lossD =  loss_fake + loss_true

            # TODO: using WGAN-GP Loss
            lossD1 = out_fake.mean() - out_true.mean()
            loss_gp = calculate_gradient_penalty(netD, images, fake, lambda_gp, device, depth=depth, alpha=alpha)
            loss_drift = torch.mean(torch.mean(out_fake ** 2) + torch.mean(out_true ** 2)) * epsilon_drift
            lossD = lossD1 + loss_gp + loss_drift

            lossD.backward()
            optimizer_D.step()
            # writer.add_scalar("lossD", lossD.item(), iter_step)
            writer.add_scalar("lossD_wasserstian", lossD1.item(), iter_step)
            writer.add_scalar("lossD_gp", loss_gp.item(), iter_step)
            writer.add_scalar("lossD_drift", loss_drift.item(), iter_step)

            # =========================
            # Update Generator
            # =========================
            optimizer_G.zero_grad()

            z = torch.randn((batch_size, dim_z), device=device)
            fake = netG(z, depth, alpha)
            out_fake = netD(fake, depth, alpha)

            # true_label = one.expand_as(out_fake)
            # lossG = criterion(out_fake,true_label)
            lossG = - out_fake.mean()
            lossG.backward()

            optimizer_G.step()

            writer.add_scalar("lossG", lossG.item(), iter_step)
            writer.add_scalar("alpha", alpha, iter_step)
            writer.add_scalar("depth", depth, iter_step)
            iter_step += 1
            bar.set_description("[iteration: %d/%d ] - [loss_G: %6.12f ] - [loss_D: %6.12f]" %(image_step, num_images, lossG.item(), lossD.item()))
            if image_step >= num_images:
                break

        with torch.no_grad():
            fake = netG(fix_noise, depth, alpha)
        # save
        save_model(netG, save_path, "netG{}x{}.pt".format(res, res))
        save_model(netD, save_path, "netD{}x{}.pt".format(res, res))
        utiler.save_images(fake, "fake{}x{}.png".format(res, res), nrow=opt.nrow)

        
endtime = time.clock()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))
