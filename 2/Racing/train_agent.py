import os
import argparse

import utils
from model import Model
from dataloader import dataloader_train_val

import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def get_loss_function(pred_mode):
    if pred_mode == "classification":
        return F.cross_entropy
    elif pred_mode == "regression":
        return F.mse_loss
    else:
        raise ValueError(f"Invalid prediction mode: {pred_mode}")

def train_epoch(model, optimizer, train_loader, opt, device):
    total_loss = 0.0
    loss_func = get_loss_function(opt.prediction_mode)

    model.train()
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1e-3)
        optimizer.step()

        total_loss += loss.item() * len(input)
        # Check if loss is decreasing
        if not opt.save_output:
            print(f"Loss = {loss.item()}")

    total_loss /= len(train_loader.dataset)
    return total_loss



def val_epoch(model, val_loader, opt, device):
    total_loss = 0.0
    loss_func = get_loss_function(opt.prediction_mode)

    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = loss_func(output, target)
            total_loss += loss.item() * len(input)

    total_loss /= len(val_loader.dataset)
    return total_loss



def train_model(
        model, optimizer, scheduler,
        train_loader, val_loader, opt
    ):
    # create result and model folders
    if not os.path.exists(opt.output_dir):
        print("Creating Folder {opt.out_dir}")
        os.mkdir(opt.output_dir)


    device = utils.get_device()
    print("Training Model")
    print(f"Device = {device}")
    print(f"Num Epochs = {opt.num_epoch}")
    print(f"Training Data Size = {len(train_loader.dataset)}")
    print(f"Validation Data Size = {len(val_loader.dataset)}")
    print(f"Model Name = {model.get_file_name()}")
    if opt.save_output:
        print("Saving model and loss logs")
    else:
        print("Not saving anything")
    if opt.save_output:
        tb_dir = os.path.join(opt.output_dir, model.title)
        if not os.path.exists(tb_dir):
            os.mkdir(tb_dir)
        logger = SummaryWriter(log_dir=tb_dir, comment=model.title)

    best_val_loss = float("inf")
    for epoch in range(opt.num_epoch):
        print(f"Starting Epoch {epoch}", flush=True)

        train_loss = train_epoch(model, optimizer, train_loader, opt, device)
        val_loss = val_epoch(model, val_loader, opt, device)
        scheduler.step(val_loss)

        if opt.save_output:
            logger.add_scalar('Loss/Train', train_loss, epoch)
            logger.add_scalar('Loss/Val', val_loss, epoch)
            if val_loss < best_val_loss:
                model.save()

        print(f"Train Loss = {train_loss}")
        print(f"Val Loss = {val_loss}")
        best_val_loss = min(best_val_loss, val_loss)

    if opt.save_output:
        logger.close()


def init_model(opt):
    device = utils.get_device()
    model = Model(opt, opt.output_dir).to(device)
    optimizer = optim.SGD(model.decoder.parameters() if opt.freeze_encoder else model.parameters(),
                    lr=1e-4, momentum=0.9, weight_decay=1e-6
                )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=4, verbose=True)

    return model, optimizer, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default="./data")
    parser.add_argument('-output_dir', type=str, default="./train_output")
    parser.add_argument('--save_output', action="store_true")
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-num_epoch', type=int, default=60)
    parser.add_argument('-num_val', type=int, default=2000)
    parser.add_argument('-N_history', type=int, default=8)
    parser.add_argument('-encoder', type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument('--freeze_encoder', action="store_true")
    parser.add_argument('-num_decoder_layers', type=int, default=1)
    parser.add_argument('-prediction_mode', type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument('--download_encoder', action="store_true")

    return parser.parse_args()

def main():
    opt = parse_args()
    model, optimizer, scheduler = init_model(opt)
    train_loader, val_loader = dataloader_train_val(opt)
    train_model(
        model, optimizer, scheduler,
        train_loader, val_loader, opt
    )

if __name__ == "__main__":
    main()
 
