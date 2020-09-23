import os, sys
import argparse
import PIL.Image as im

from math import log2
from tqdm import tqdm

from utils import test_and_add_postfix_dir

tores = lambda x: int(log2(x))
fromres = lambda x: 2 ** x

def preprocess(path, dest_path, start_res=4, end_res=1024):
    # ilist = os.listdir(path)[:20] # for test
    ilist = os.listdir(path)

    dest_path = test_and_add_postfix_dir(dest_path)
    path = test_and_add_postfix_dir(path)

    sres = tores(start_res)
    eres = tores(end_res)

    res_paths = []
    for i in range(sres, eres + 1):
        name = dest_path + "data_{}x{}".format(fromres(i), fromres(i)) + os.sep
        if os.path.exists(name):
            os.removedirs(name)
        os.makedirs(name)
        res_paths.append(name)

    for ifile in tqdm(ilist):
        image = im.open(path + ifile)
        for i, res in enumerate(range(sres, eres + 1)):
            dimage = image.resize((fromres(res), fromres(res)), resample=im.BICUBIC) # TODO: optional interpolation mode
            dimage.save(res_paths[i] + ifile)
            dimage.close()
    print("OK")


def get_options():
    parser = argparse.ArgumentParser()
#    parser.add_argument("--path", required=True, type=str, help="source image path")
#    parser.add_argument("--dpath", type=str, default="./save/", help="output image path")
    parser.add_argument("--im", type=int, default=im.BICUBIC, help="interpolation mode")
    parser.add_argument("--sr", type=int, default=4, help="start resolution")
    parser.add_argument("--er", type=int, default=1024, help="end resolution")
    opt = parser.parse_args()
    return opt

def main():
    opt = get_options()
    setattr(opt, "path", "/home/victorchen/workspace/Aristotle/StyleGAN_PyTorch/FFHQ"
)
    setattr(opt, "dpath", "./dataset/ffhq/")
    preprocess(opt.path, opt.dpath, opt.sr, opt.er)

if __name__ == '__main__':
    main()
