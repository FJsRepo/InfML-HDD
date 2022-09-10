import os
import random
import argparse
import cv2
import numpy as np
import torch
from lib.config import Config


def save_laplace_core(model, exp_root, epoch, layer):
    model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])
    if model == 500:
        print(model.state_dict()["model.layer1.0.conv2.weight"][0, 0])
    mean_list_0 = 0
    mean_list_1 = 0
    mean_list_2 = 0
    for key in model.state_dict().keys():
        if (layer in key) and ('conv2' in key):
            number_of_kernel, number_of_kernel_layers, _, _ = model.state_dict()[key].shape
            temp = 0
            count = 0
            for index in range(number_of_kernel):
                for index2 in range(number_of_kernel_layers):
                    temp = temp + model.state_dict()[key][index, index2][1,1].cpu().numpy()
                    count = count + 1
            mean = temp / count
            if '.0.conv2.' in key:
                mean_list_0 = mean
            elif '.1.conv2.' in key:
                mean_list_1 = mean
            elif '.2.conv2.' in key:
                mean_list_2 = mean
            else:
                print("Wrong key!")
            print("model:", epoch, ":", key, "finished", number_of_kernel, number_of_kernel_layers)
    return mean_list_0, mean_list_1, mean_list_2

def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = cfg.get_model().to(device)
    total_epochs = cfg["epochs"]
    layer = 'layer1'
    for index_epoch in range(1, total_epochs+1):
        mean_list_0, mean_list_1, mean_list_2 = save_laplace_core(model, exp_root, index_epoch, layer)
        with open('layer1_0_laplace_mean_core.txt', 'a') as f0:
            f0.write(str(mean_list_0) + '\n')
        with open('layer1_1_laplace_mean_core.txt', 'a') as f1:
            f1.write(str(mean_list_1) + '\n')
        with open('layer1_2_laplace_mean_core.txt', 'a') as f2:
            f2.write(str(mean_list_2) + '\n')
