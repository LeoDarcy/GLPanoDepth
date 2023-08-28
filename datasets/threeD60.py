import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from .util import Equirec2Cube

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class ThreeD60(data.Dataset):

    def __init__(self, root_dir, list_file, height=256, width=512, is_training=False):
        """
        Args:
            root_dir (string): Directory of the 3D60 Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.w = width
        self.h = height
        self.is_training = is_training

        self.max_depth_meters = 10.0

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])

        assert os.path.isfile(rgb_name) == True, "No such rgb file " + str(rgb_name)
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h))

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        assert os.path.isfile(depth_name) == True, "No such depth file " + str(depth_name)
        gt_depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth[gt_depth>self.max_depth_meters] = self.max_depth_meters + 1
        
        cube_rgb = self.e2c.run(rgb)
        aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        
        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["cube_rgb"] = cube_rgb
        inputs["normalized_cube_rgb"] = torch.stack(torch.split(self.normalize(cube_rgb), self.h // 2, -1), dim=0)
        
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters) & ~torch.isnan(inputs["gt_depth"]))
        return inputs
