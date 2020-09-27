import os, sys
import argparse
import PIL.Image as im

from math import log2
from tqdm import tqdm

# import threading
import multiprocessing

from utils import test_and_add_postfix_dir

"""
Preprocess the images, transform original images to different resolution as to speed up 
loading in DataLoader.
"""

tores = lambda x: int(log2(x))
fromres = lambda x: 2 ** x


class ProcessThread(multiprocessing.Process):
    def __init__(self, ilist, path, res_paths, start_res=4, end_res=1024, override=False):
        multiprocessing.Process.__init__(self)
        self.ilist = ilist
        self.path = path
        self.res_paths = res_paths
        self.start_res = start_res
        self.end_res = end_res
        self.override = override

    def run(self):

        sres = tores(self.start_res)
        eres = tores(self.end_res)

        for ifile in tqdm(self.ilist):
            image = im.open(self.path + ifile)
            for i, res in enumerate(range(sres, eres + 1)):
                save_file_path = self.res_paths[i] + ifile
                if not self.override and os.path.exists(save_file_path):
                    continue
                dimage = image.resize((fromres(res), fromres(res)), resample=im.BICUBIC) # TODO: optional interpolation mode
                dimage.save(save_file_path)
                dimage.close()

        print("OK")


def mulitiprocess(path, dest_path, num_workers=4, start_res=4, end_res=1024, override=False):
    # ilist = os.listdir(path)[:20] # for test
    ilist = [ image for image in os.listdir(path) if image.split(".")[-1] in ["png", "jpg", "jpeg"]] # load and filter image names

    dest_path = test_and_add_postfix_dir(dest_path)
    path = test_and_add_postfix_dir(path)

    sres = tores(start_res)
    eres = tores(end_res)

    # make dirs
    res_paths = []
    for i in range(sres, eres + 1):
        name = dest_path + "data_{}x{}".format(fromres(i), fromres(i)) + os.sep
        if os.path.exists(name) and override:
            os.removedirs(name)
            os.makedirs(name)
        elif not os.path.exists(name):
            os.makedirs(name)
        res_paths.append(name)

    # start threads
    threads = []
    part_size = len(ilist) // num_workers + 1
    for i in range(num_workers):
        target_list = ilist[i * part_size: min((i+1) * part_size, len(ilist))]
        thread = ProcessThread(target_list, path, res_paths, start_res, end_res)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    print("OK")

def preprocess(path, dest_path, start_res=4, end_res=1024, override=False):
    # ilist = os.listdir(path)[:20] # for test
    ilist = os.listdir(path)

    dest_path = test_and_add_postfix_dir(dest_path)
    path = test_and_add_postfix_dir(path)

    sres = tores(start_res)
    eres = tores(end_res)

    res_paths = []
    for i in range(sres, eres + 1):
        name = dest_path + "data_{}x{}".format(fromres(i), fromres(i)) + os.sep
        if os.path.exists(name) and override:
            os.removedirs(name)
            os.makedirs(name)
        elif not os.path.exists(name):
            os.makedirs(name)
        res_paths.append(name)

    for ifile in tqdm(ilist):
        # check the extension
        ext = ifile.split(".")[-1]
        if ext not in ["png", "jpg", "jpeg"]:
            continue
        image = im.open(path + ifile)
        for i, res in enumerate(range(sres, eres + 1)):
            save_file_path = res_paths[i] + ifile
            if not override and os.path.exists(save_file_path):
                continue
            dimage = image.resize((fromres(res), fromres(res)), resample=im.BICUBIC) # TODO: optional interpolation mode
            dimage.save(save_file_path)
            dimage.close()
    print("OK")


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="source image path")
    parser.add_argument("--dpath", type=str, default="./save/", help="output image path")
    parser.add_argument("--im", type=int, default=im.BICUBIC, help="interpolation mode")
    parser.add_argument("--sr", type=int, default=4, help="start resolution")
    parser.add_argument("--er", type=int, default=1024, help="end resolution")
    parser.add_argument("--num-workers", type=int, default=1, help="number of threads")
    parser.add_argument("--override", type=bool, default=False, help="override file with same name or not")
    opt = parser.parse_args()
    
    # self defined
    return opt

def main():
    opt = get_options()
    if opt.num_workers <= 1:
        preprocess(opt.path, opt.dpath, opt.sr, opt.er)
    else:
        mulitiprocess(opt.path, opt.dpath, opt.num_workers, opt.sr, opt.er)

if __name__ == '__main__':
    main()
