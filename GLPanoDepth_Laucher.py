import os
import numpy as np
import time
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import datasets
from models import TwoBranch, TransformerNet, ModelLoadTransformer
from utils.metrics import compute_depth_metrics, Evaluator
from utils.losses import BerhuLoss

class GLPanoDepth:
    def __init__(self, args):
        self.settings = args

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in self.settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        # data
        datasets_dict = {"3d60": datasets.ThreeD60,
                         "stanford2d3d": datasets.Stanford2D3D,
                         "matterport3d": datasets.Matterport3D}
        self.dataset = datasets_dict[self.settings.dataset]
        self.settings.cube_w = self.settings.height//2

        fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

        train_file_list = fpath.format(self.settings.dataset, "train")
        val_file_list = fpath.format(self.settings.dataset, "val")

        train_dataset = self.dataset(self.settings.data_path, train_file_list, self.settings.height, self.settings.width, is_training=True)
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        num_train_samples = len(train_dataset)

        val_dataset = self.dataset(self.settings.data_path, val_file_list, self.settings.height, self.settings.width, is_training=False)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        #BranchLoader
        Model_dict = {"TwoBranch":TwoBranch, "TransformerNet":TransformerNet}
        Net = Model_dict[self.settings.net]

        self.model = Net(image_height=self.settings.height, image_width=self.settings.width)
        
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        
        self.parameters_to_train = list(self.model.parameters())
        
        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()
        
        

        self.compute_loss = BerhuLoss(threshold=self.settings.berhuloss)
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        
        print("Init successfully! Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

    def train(self):
        """Run the entire training pipeline
        """
        
        
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()
        
        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        for batch_idx, inputs in enumerate(pbar):
            outputs, losses = self.forward_batch(inputs)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

    def forward_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb", "cube_rgb"]:
                inputs[key] = ipt.to(self.device)

        losses = {}

        equi_inputs = inputs["normalized_rgb"]
        cube_input = inputs["normalized_cube_rgb"]
        trans_output, our_outputs = self.model(equi_inputs, cube_input)
        outputs = our_outputs

        
        our_loss = self.compute_loss(inputs["gt_depth"],
                                           our_outputs,
                                           inputs["val_mask"])
        losses["loss"] = our_loss
        
        return outputs, losses

    def validate(self):
        """
        Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.forward_batch(inputs)
                pred_depth = outputs

                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth, mask)

        for i, key in enumerate(self.evaluator.metrics.keys()):
            print("Validation Results: ", key, self.evaluator.metrics[key], self.evaluator.metrics[key].avg)
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses


    def log(self, mode, inputs, outputs, losses):
        """
        Write log for each epoch
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.epoch)

        pred_depth = outputs
        
        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.epoch)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.epoch)
            writer.add_image("pred_depth/{}".format(j),
                             pred_depth[j].data/pred_depth[j].data.max(), self.epoch)

    def save_model(self):
        """
        Save model
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net

        torch.save(to_save, save_path)

    def load_model(self):
        """
        Load model
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("Loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        

    def loop_dataset(self):
        """
        Check dataset
        """
        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(1))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                for key, ipt in inputs.items():
                    if key not in ["rgb", "cube_rgb"]:
                        inputs[key] = ipt.to(self.device)
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                print("In loop dataset function")
                print("gt_depth ", gt_depth.shape)
                print("mask", mask.shape)
        
        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(1))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                for key, ipt in inputs.items():
                    if key not in ["rgb", "cube_rgb"]:
                        inputs[key] = ipt.to(self.device)
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                print("In loop dataset function")
                print("gt_depth ", gt_depth.shape)
                print("mask", mask.shape)
        return False
                

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook