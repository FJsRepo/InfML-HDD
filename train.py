import os
import sys
import random
import shutil
import logging
import argparse
import subprocess
from time import time

import numpy as np
import torch

from test import test
from lib.config import Config

def train(model, train_loader, exp_dir, cfg, val_loader, train_state=None):
    # Get initial train state
    optimizer = cfg.get_optimizer(model.parameters())
    scheduler = cfg.get_lr_scheduler(optimizer)
    starting_epoch = 1

    if train_state is not None:
        model.load_state_dict(train_state['model'])
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['lr_scheduler'])
        starting_epoch = train_state['epoch'] + 1
        scheduler.step(starting_epoch)

    # Train the model
    criterion_parameters = cfg.get_loss_parameters()
    criterion = model.loss
    total_step = len(train_loader)
    ITER_LOG_INTERVAL = cfg['iter_log_interval']
    ITER_TIME_WINDOW = cfg['iter_time_window']
    MODEL_SAVE_INTERVAL = cfg['model_save_interval']
    t0 = time()
    total_iter = 0
    iter_times = []
    logging.info("Starting training.")
    for epoch in range(starting_epoch, num_epochs + 1):
        epoch_t0 = time()
        logging.info("Beginning epoch {}".format(epoch))
        accum_loss = 0
        for i, (images, labels, img_idxs) in enumerate(train_loader):
            total_iter += 1
            iter_t0 = time()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, epoch=epoch)
            loss, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            accum_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_times.append(time() - iter_t0)
            if len(iter_times) > 100:
                iter_times = iter_times[-ITER_TIME_WINDOW:]
            if (i + 1) % ITER_LOG_INTERVAL == 0:
                loss_str = ', '.join(
                    ['{}: {:.4f}'.format(loss_name, loss_dict_i[loss_name]) for loss_name in loss_dict_i])
                logging.info("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({}), s/iter: {:.4f}, lr: {:.1e}".format(
                    epoch,
                    num_epochs,
                    i + 1,
                    total_step,
                    accum_loss / (i + 1), #此loss是累计损失的平均值
                    loss_str,
                    np.mean(iter_times),
                    optimizer.param_groups[0]["lr"],
                ))
        logging.info("Epoch time: {:.4f}".format(time() - epoch_t0))
        if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == num_epochs:
            model_path = os.path.join(exp_dir, "models", "model_{:03d}.pt".format(epoch))
            save_train_state(model_path, model, optimizer, scheduler, epoch)
        scheduler.step()
    logging.info("Training time: {:.4f}".format(time() - t0))
    return model


def save_train_state(path, model, optimizer, lr_scheduler, epoch):
    train_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }

    torch.save(train_state, path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PolyLaneNet")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training")#有action的时候此键值对应true，为bool类型
    parser.add_argument("--validate", action="store_true", help="Validate model during training")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")

    return parser.parse_args()


def setup_exp_dir(exps_dir, exp_name, cfg_path):
    dirs = ["models"]
    exp_root = os.path.join(exps_dir, exp_name)
    for dirname in dirs:
        os.makedirs(os.path.join(exp_root, dirname), exist_ok=True)
    os.makedirs(os.path.join(exp_root, "pictures"), exist_ok=True)
    shutil.copyfile(cfg_path, os.path.join(exp_root, 'config.yaml'))
    return exp_root

def get_exp_train_state(exp_root):
    models_dir = os.path.join(exp_root, "models")
    models = os.listdir(models_dir)
    last_epoch, last_modelname = sorted(
        [(int(name.split("_")[1].split(".")[0]), name) for name in models],
        key=lambda x: x[0],
    )[-1]
    train_state = torch.load(os.path.join(models_dir, last_modelname))

    return train_state


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def save_laplace_core(model, exp_root, epoch, layer):
    model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])
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


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up experiment
    if not args.resume:
        exp_root = setup_exp_dir(cfg['exps_dir'], args.exp_name, args.cfg)
    else:
        exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))
    # Set log
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "log.txt")),#日志输出到文件
            logging.StreamHandler(),
        ],
    )
    sys.excepthook = log_on_exception
    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Get data sets
    train_dataset = cfg.get_dataset("train")
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    model = cfg.get_model().to(device)

    train_state = None
    if args.resume:
        train_state = get_exp_train_state(exp_root)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    # calculate mean and std
    # data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=len(train_dataset),
    #                                            shuffle=True,
    #                                            num_workers=8)
    # data = next(iter(data_loader))
    # mean = data[0].mean() #tensor(-0.0138)
    # std = data[0].std() #tensor(0.9587)

    if args.validate:
        val_dataset = cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,#验证数据不打乱
                                                 num_workers=8)
    # Train regressor
    try:
        model = train(
            model,
            train_loader,
            exp_root,
            cfg,
            val_loader=val_loader if args.validate else None,
            train_state=train_state,
        )
    except KeyboardInterrupt:
        logging.info("Training session terminated.")
    total_epoch = cfg["epochs"]
    if cfg['backup'] is not None:
        subprocess.run(['rclone', 'copy', exp_root, '{}/{}'.format(cfg['backup'], args.exp_name)])

