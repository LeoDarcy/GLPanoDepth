import os
import numpy as np
import time
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import argparse
from GLPanoDepth_Laucher import GLPanoDepth


torch.manual_seed(100)
torch.cuda.manual_seed(100)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--data_path", default="/data/bjy/DataSet/3D60/Center/", type=str, help="path to dataset")
    parser.add_argument("--dataset", default="3d60", choices=["3d60", "matterport3d", "stanford2d3d"], type=str, help="which dataset to train")
    parser.add_argument("--transformer_path", type=str, help="path to load pertrained CViT")
    
    #network settings
    parser.add_argument("--net", type=str, default="TwoBranch", choices=["SphereNet", "NormalNet", "TwoBranch"], help="choose branch")
    parser.add_argument("--model_name", type=str, default="GLPanoDepth", help="model name")
    parser.add_argument("--height", type=int, default=256, help="input image height")
    parser.add_argument("--width", type=int, default=512, help="input image width")

    #loss settings
    parser.add_argument("--berhuloss", type=float, default=0.2, help="berhu loss threhold")
    parser.add_argument("--learning_rate", type=float, default=1*1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    
    #system settings
    parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument("--gpu_devices", type=int, nargs="+", default=[3], help="available gpus")

    # loading and logging settings
    parser.add_argument("--load_weights_dir", type=str, help="path to trained model")
    parser.add_argument("--log_dir", type=str, default="/data/bjy/DepthEstimation/Code/GLPanoDepthLog/", help="path to log")
    
    # network setting
    args = parser.parse_args()
    model = GLPanoDepth(args)
    model.train()