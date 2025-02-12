from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import sys
import time
from datetime import datetime

sys.path.append("code/python/")

from Utils import set_layer_mode
from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

def binary_hingeloss(yhat, y, b=128):
    # print("BINHINGE")

    # print("yhat", yhat.mean(dim=1)) # output <=> predictions
    # print("y", y)                   # target <=> ground truth labels
    y_enc = 2 * torch.nn.functional.one_hot(y, yhat.shape[-1]) - 1.0 

    # l_MHL = max{0, (b - y_enc * yhat)}
    l = (b - y_enc * yhat).clamp(min=0) 
    
    return l.mean(dim=1) / b


class Clippy(torch.optim.Adam):
    def step(self, closure=None):
        loss = super(Clippy, self).step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp(-1,1)
        return loss

class Criterion:
    def __init__(self, method, name, param=None):
        self.method = method
        self.param = param
        self.name = name
    def applyCriterion(self, output, target):
        if self.param is not None:
            return self.method(output, target, self.param)
        else:
            return self.method(output, target)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    set_layer_mode(model, "train") # propagate informaton about training to all layers

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        # loss = F.nll_loss(output, target)

        # print(model.traincriterion.name)
        loss = model.traincriterion.applyCriterion(output, target).mean()

        ### Original SPICE-Torch
        # if model.name == "ResNet18":
        #     loss = model.traincriterion.applyCriterion(output, target).mean()
        # else:
        #     criterion = nn.CrossEntropyLoss(reduction="none")
        #     loss = criterion(output, target).mean()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, pr=1):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # print(model.testcriterion.name)
            test_loss += model.testcriterion.applyCriterion(output, target).mean()  # sum up batch loss

            ### Original SPICE-Torch
            # if model.name == "ResNet18":
            #     test_loss += model.testcriterion.applyCriterion(output, target).mean()  # sum up batch loss
            # else:
            #     test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    if pr is not None:
        print('\nAccuracy: {:.2f}%\n'.format(
            100. * correct / len(test_loader.dataset)))

    accuracy = 100. * (correct / len(test_loader.dataset))

    return accuracy

def test_error(model, device, test_loader, perror):
    
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    if model.name == "ResNet18":
        for block in model.children():
            if isinstance(block, (QuantizedActivation, QuantizedConv2d)):
                if block.error_model is not None:
                    block.error_model.updateErrorModel(perror)
            if isinstance(block, (QuantizedLinear)):
                if block.error_model is not None:
                    block.error_model.updateErrorModel(perror)
            if isinstance(block, nn.Sequential):
                for layer in block.children():
                    # if layer.error_model is not None:
                    #     layer.error_model.updateErrorModel(perror)
                    for inst in layer.children():
                        if isinstance(inst, (QuantizedLinear, QuantizedConv2d)):
                            if inst.error_model is not None:
                                inst.error_model.updateErrorModel(perror)
                        if isinstance(inst, nn.Sequential):
                            for shortcut_stuff in inst.children():
                                if isinstance(shortcut_stuff, (QuantizedLinear, QuantizedConv2d)):
                                    if shortcut_stuff.error_model is not None:
                                        shortcut_stuff.error_model.updateErrorModel(perror)
    else:
        # update perror in every layer
        for layer in model.children():
            if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
                if layer.error_model is not None:
                    layer.error_model.updateErrorModel(perror)

    print("Error rate: ", perror)
    
    # start_time = time.perf_counter()
    accuracy = test(model, device, test_loader)
    # end_time = time.perf_counter()
    # elapsed_time = end_time-start_time
    # minutes, seconds = divmod(elapsed_time, 60)
    # formatted_time = f"{int(minutes):02}:{seconds:05.2f}"
    # print(f"Elapsed time: {formatted_time}")

    # reset error models
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if layer.error_model is not None:
                layer.error_model.resetErrorModel()

    return accuracy
