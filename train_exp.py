import time, os
import gc
gc.collect()
from math import log2

import torch
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
    TensorImageUtils, save_model, partial_generate

from WGANLoss import calculate_gradient_penalty
from ProGAN import ProGenerator, ProDiscriminator

tores = lambda x: int(log2(x))
fromres = lambda x: 2 ** x
#------------------ config by options ---------------------

# persistence related configuration
save_path = test_and_add_postfix_dir("tmp" + os.sep + "save_" + currentTime()) 
test_and_make_dir(save_path)
writer = SummaryWriter(save_path)
utiler = TensorImageUtils(save_path)

# wait here
for i in range(45):
    time.sleep(60)
    print("Waiting %2d minutes" %(i+1))

# training related configuration
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

# model parameters
min_res = tores(4)
max_res = opt.max_depth
max_resolution = fromres(max_res)

# start_res, end_res 控制需要训练的 blocks.
start_res = max(min_res, tores(opt.start_resolution))
end_res = min(max_res, tores(opt.end_resolution))

max_num_blocks = max_res - 2
depths = [-1] + [ i // 2 for i in range(max_num_blocks * 2)] # predefined depths
dynamic_batch_size = [32, 32, 32, 32, 16, 10, 6, 3, 2] # predefined batch size to avoid excceeding gpu memory

# define model here
netD = ProDiscriminator(resolution=max_resolution)
netG = ProGenerator(resolution=max_resolution)

# load pre-trained model
if opt.resume:
    model_path = test_and_add_postfix_dir(opt.model_dir)
    resume_res = opt.resume_resolution
    G_name = "netG{}x{}.pt".format(resume_res, resume_res)
    D_name = "netD{}x{}.pt".format(resume_res, resume_res)
    netD.load_state_dict(torch.load(model_path + D_name, map_location="cpu"))
    netG.load_state_dict(torch.load(model_path + G_name, map_location="cpu"))

if opt.cuda:
    netD.cuda()
    netG.cuda()

# trained resolution
first_stablize_train = True

optimizer_D = optim.Adam(netD.parameters(), lr=opt.lrD, betas=opt.adam_betas)
optimizer_G = optim.Adam(netG.parameters(), lr=opt.lrG, betas=opt.adam_betas)

#------------------ Training -------------------------

print("Start Training, using {}".format(device))
starttime = time.clock()

# original GAN Loss
criterion = torch.nn.BCEWithLogitsLoss()
zero = torch.tensor(0, dtype=torch.float).to(device)
one = torch.tensor(1, dtype=torch.float).to(device)

fix_noise = torch.randn((max(dynamic_batch_size), opt.dim_z), dtype=torch.float32).to(device)

opt.num_images = opt.num_images * 1000

# progressive training.
iter_step = 0
for i, depth in enumerate(depths):

    stablize = (i % 2 == 0)

    res = 2 ** (depth + 3)
    if res < start_res or res > end_res: continue # train in resolution range
    if opt.train_stablize_first and res == start_res and not stablize: continue # start from resolution and stabliaztoin mode

    print("depth:{} - res:{}x{} - training mode:{}".format(
        depth, res, res, "stablize" if stablize else "progressive"
    ))

    # ============ redefine dataset ========================
    bs = dynamic_batch_size[depth + 1]
    opt.batch_size = bs
    opt.input_size = res
    data = choose_dataset(opt) # change batch_size and input_size
    print("Using batch_size: {}".format(bs))

    # =========== training =================================
    image_step = 0
    while image_step < opt.num_images:
        bar = tqdm(data)
        for j, batch in enumerate(bar):
            images, _ = batch
            if opt.cuda:
                images = images.cuda()

            batch_size = images.size(0)
            image_step += batch_size

            if stablize:
                alpha = 1.0
            else:
                knum = opt.num_images // 1000
                alpha = (min(image_step, opt.num_images) // 1000) / knum # for each alpha train for 1000 images

            # =========================
            # Update Discriminator 
            # =========================
            optimizer_D.zero_grad()

            z = torch.randn((batch_size, opt.dim_z), device=device)
            fake = netG(z, depth, alpha)
            out_fake = netD(fake.detach(), depth, alpha)
            out_true = netD(images, depth, alpha)

            # GAN Loss
            # true_label = one.expand_as(out_true)
            # fake_label = zero.expand_as(out_fake)
            
            # loss_fake = criterion(out_fake, fake_label)
            # loss_true = criterion(out_true, true_label)
            # lossD =  loss_fake + loss_true

            # WGAN-GP Loss
            lossD1 = out_fake.mean() - out_true.mean()
            loss_gp = calculate_gradient_penalty(netD, images, fake, opt.lambda_gp, device, depth=depth, alpha=alpha)
            loss_drift = torch.mean(torch.mean(out_fake ** 2) + torch.mean(out_true ** 2)) * opt.epsilon_drift
            lossD = lossD1 + loss_gp + loss_drift

            lossD.backward()
            optimizer_D.step()
            writer.add_scalar("lossD", lossD.item(), iter_step)
            writer.add_scalar("lossD_wasserstian", lossD1.item(), iter_step)
            writer.add_scalar("lossD_gp", loss_gp.item(), iter_step)
            writer.add_scalar("lossD_drift", loss_drift.item(), iter_step)

            # =========================
            # Update Generator
            # =========================
            optimizer_G.zero_grad()

            z = torch.randn((batch_size, opt.dim_z), device=device)
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
            bar.set_description("[iteration: %d/%d ] - [loss_G: %6.12f ] - [loss_D: %6.12f]" %(image_step, opt.num_images, lossG.item(), lossD.item()))
            if image_step >= opt.num_images:
                break

            if (image_step + 1) % 30000 == 0:
                fake = partial_generate(netG, fix_noise, opt.batch_size, depth=depth, alpha=1.0)
                utiler.save_images(fake, "fake{}x{}_{}.png".format(res, res, image_step), nrow=opt.nrow)

        with torch.no_grad():
            fake = partial_generate(netG, fix_noise, opt.batch_size, depth=depth, alpha=1.0)
        # save
        save_model(netG, save_path, "netG{}x{}.pt".format(res, res))
        save_model(netD, save_path, "netD{}x{}.pt".format(res, res))
        utiler.save_images(fake, "fake{}x{}.png".format(res, res), nrow=opt.nrow)
        save_model(optimizer_G, save_path, "optimG{}x{}.pt".format(res, res))
        save_model(optimizer_D, save_path, "optimD{}x{}.pt".format(res, res))

        
endtime = time.clock()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))
