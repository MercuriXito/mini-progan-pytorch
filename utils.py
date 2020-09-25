import os, sys, time, json
from torchvision.utils import make_grid, save_image
from torch import save, load
import torch
from math import ceil

import matplotlib.pyplot as plt

def test_and_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def currentTime():
    return time.strftime("%H_%M_%S", time.localtime())

def test_and_add_postfix_dir(root):
    seplen = len(os.sep)
    if root[-seplen:] != os.sep:
        return root + os.sep
    return root

def json_dump(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)

def save_opt(root, opt):
    json_dump(opt._get_kwargs(), root + "config.json")

def save_images(images, root, filename, nrow = 8):
    save_image(images, root + filename, nrow=nrow, normalize=True, range=(-1,1))

def save_model(model, root, filename):
    save(model.state_dict(), root + filename)

def show_images(images, nrow = 8, img_range = (-1, 1)):
    grid = make_grid(
                    images, nrow=nrow, range=img_range
                ).cpu().detach().numpy().transpose((1,2,0))
    return grid

#============== visualize utils ================

class TensorImageUtils:
    """Base Class of Tensor-Image utils functions including showing and saving the result images,
    `prepreocess_tensor` function is used to preprocess images before showing and saving"""
    def __init__(self, root = ".", img_range=(-1, 1), normalize=True, preprocess_func = None):
        self.root = test_and_add_postfix_dir(root)
        self.img_range = img_range
        self.normalize = normalize
        if preprocess_func is None:
            self.preprocessor = self.preprocess_tensor
        else:
            self.preprocessor = preprocess_func

    def preprocess_tensor(self, images_tensor, *args, **kws):
        """ Default preprocessor, return tensor directly
        """
        return images_tensor

    def tensor2arr(self, images, nrow = 8):
        timages = self.preprocessor(images.detach())
        grid = make_grid(
                timages, nrow=nrow, normalize=self.normalize, range=self.img_range).cpu().detach().numpy().transpose((1,2,0))
        return grid

    def plot_show(self, images, nrow = 8, figsize=(15, 15), is_rgb=False):
        fig = plt.figure(figsize=figsize)
        target_cmap = plt.cm.rainbow if is_rgb else plt.cm.binary_r
        arr = self.tensor2arr(images, nrow)
        plt.imshow(arr, cmap=target_cmap)

    def save_images(self, images, filename, nrow=8):
        images = self.preprocessor(images)
        save_image(images, self.root + filename,
                   nrow=nrow, normalize=self.normalize, range=self.img_range)


def partial_generate(netG, z, batch_size, **kws):
    """ generate images with netG and z, but use `batch_size` number of images in forward
    for avoiding excceeding the gpu memory.
    """
    total_num = z.size(0)
    num = 0
    images = []
    while(num < total_num):
        part_z = z[num * batch_size: min(total_num, (num + 1) * batch_size), :]
        if part_z.size(0) == 0: break
        with torch.no_grad():
            part_image = netG(part_z, **kws)
        images.append(part_image)
        num += 1
    if len(images) == 1:
        return images[0]
    else:
        return torch.cat(images, dim=0)


if __name__ == "__main__":
    pass
