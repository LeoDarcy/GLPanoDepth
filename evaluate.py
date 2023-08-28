import os
import argparse
import tqdm
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torch.utils.data import DataLoader
from models import TwoBranch, TransformerNet,ModelLoadTransformer
import datasets
from utils.metrics import Evaluator
parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="/data/bjy/DataSet/3D60/Center/", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="3d60", choices=["3d60", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to evaluate on.")
parser.add_argument("--transformer_path",  type=str, help="path to the dataset.")

parser.add_argument("--load_weights_dir", type=str, default="/data/bjy/DepthEstimation/Code/GLPanoDepthLog/TwoBranchTransformer_1110_cube/models/weights_0", help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")


settings = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model.pth")
    model_dict = torch.load(model_path)
    # data
    datasets_dict = {"3d60": datasets.ThreeD60,
                     "stanford2d3d": datasets.Stanford2D3D,
                     "matterport3d": datasets.Matterport3D}
    dataset = datasets_dict[settings.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    test_file_list = fpath.format(settings.dataset, "test")

    test_dataset = dataset(settings.data_path, test_file_list,
                           model_dict['height'], model_dict['width'], is_training=False)
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {"TransformerNet": TransformerNet,  "TwoBranch":TwoBranch}
    Net = Net_dict[model_dict['net']]
    model = Net(image_height=model_dict['height'], image_width=model_dict['width'])
    model = torch.nn.DataParallel(model)
    model.to(device)
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_dict.items() if k in model_state_dict}
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    
    model.eval()
    # print(model)

    
    evaluator = Evaluator()
    evaluator.reset_eval_metrics()
    
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            equi_inputs = inputs["normalized_rgb"].to(device)
            cube_input = inputs["normalized_cube_rgb"].to(device)
            
            #print(p[0].shape, p)
            p, outputs = model(equi_inputs, cube_input)
            pred_depth = (outputs.detach()).cpu()
            
            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]

            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])
            

    evaluator.print(load_weights_folder)


if __name__ == "__main__":
    main()