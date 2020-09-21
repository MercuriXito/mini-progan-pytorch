"""
options for ProGAN
"""

import argparse
from data import get_mnist, get_cifar10, get_fashion, get_svhn, get_unlabeled_celebA, get_folder_dataset

def get_normal_options(parser):
    # parameters in training
    parser.add_argument("--lrD", default=2e-4, type=float, help="Learning rate of Discriminator")
    parser.add_argument("--lrG", default=2e-4, type=float, help="Learning rate of Generator")
    parser.add_argument("--epochs", default=10, type=int, help="Total epochs in training")
    parser.add_argument("--save-epoch-interval", default=1, type=int, help="interval of epoch for saving model")
    parser.add_argument("--cuda", default=True, type=bool, help="using cuda")
    parser.add_argument("--adam-betas", default=(0, 0.99), type=tuple, help="betas of adam optimizer")

    # parameters of dataset
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--data-name", default="MNIST", type=str, help="dataset name")
    parser.add_argument("--input-size", default=(64,64), type=tuple, help="output image size") # currently cannot be changed
    parser.add_argument("--nrow", default=16, type=int, help="number of rows when showing batch of images")
    parser.add_argument("--data-path", default=".", type=str, help="path of dataset")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers in dataloader")

    # misc 
    parser.add_argument("--test", default=False, type=bool, help="train one epoch for test")
    parser.add_argument("--board", default=True, type=bool, help="using tensorboard to record") # temporay use tensorboard as default
    
    return parser


def get_resume_options(parser):
    parser.add_argument("--resume", action="store_true", help="resuem training")
    parser.add_argument("--model-dir", default="null", type=str, help="folder which saves netG.pth and netD.pth")
    return parser


def get_options():
    parser = argparse.ArgumentParser()
    parser = get_normal_options(parser)
    return parser.parse_args()


def get_progan_options():
    parser = argparse.ArgumentParser()
    parser = get_normal_options(parser)
    parser = get_resume_options(parser)

    parser.add_argument("-res", "--resolution", default=256, type=int, help="input resolution")
    opt = parser.parse_args()

    # 懒得写那么多命令行参数，就直接在这改8.
    opt.batch_size = 4
    opt.data_name = "folder"
    opt.data_path = "/home/victorchen/workspace/Aristotle/StyleGAN_PyTorch/FFHQ"
    return opt


def choose_dataset(opt):
    """ choose dataset
    """
    data_name = opt.data_name
    if data_name == "MNIST":
        setattr(opt, "data_path", "/home/victorchen/workspace/Venus/torch_download/MNIST")
        setattr(opt, "in_channels", 1)
        data = get_mnist(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    elif data_name == "cifar10":
        setattr(opt, "data_path", "/home/victorchen/workspace/Venus/torch_download/")
        setattr(opt, "in_channels", 3)
        data = get_cifar10(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    elif data_name == "fashion":
        setattr(opt, "data_path", "/home/victorchen/workspace/Venus/torch_download/FashionMNIST")
        setattr(opt, "in_channels", 1)
        data = get_fashion(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    elif data_name == "svhn":
        setattr(opt, "data_path", "/home/victorchen/workspace/Venus/torch_download/svhn")
        setattr(opt, "in_channels", 3)
        data = get_svhn(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    elif data_name == "unlabeled_celeba":
        setattr(opt, "data_path", "/home/victorchen/workspace/Venus/celebA/images")
        setattr(opt, "in_channels", 3)
        data = get_unlabeled_celebA(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    elif data_name == "folder":
        data = get_folder_dataset(opt.data_path, opt.batch_size, opt.num_workers, opt.input_size)
    else:
        raise NotImplementedError("Not implemented dataset: {}".format(data_name))
    return data


class _MetaOptions:
    """ options-like object
    """
    def __str__(self):
        return ";".join(["{}:{}".format(key,val) for key, val in self.__dict__.items()])
    
    @staticmethod
    def kws2opts(**kws):
        """ Recursively convert all keyword input to option like object.
        """
        return _MetaOptions.dict2opts(kws)

    @staticmethod
    def dict2opts(d: dict):
        """ Recursively convert dict to option like object.
        """
        o = _MetaOptions()
        def _parse(obj, dt: dict):
            for key, val in dt.items():
                if not isinstance(key, str):
                    raise AttributeError("Not allowed key in dict with type:{}".format(type(key)))
                if isinstance(val, dict):
                    t = _MetaOptions()
                    setattr(obj, key, t)
                    _parse(t, val)
                else:
                    setattr(obj, key, val)
            return obj
        return _parse(o, d)


if __name__ == "__main__":
    opt = _MetaOptions.kws2opts(name="test", lr=1e-3, epochs=20)
    print(opt.name)
    print(opt.lr, opt.epochs)
