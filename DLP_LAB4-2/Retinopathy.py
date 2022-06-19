from dataloader import RetinopathyLoader
from torch import Tensor, device, cuda, no_grad, load, save
from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
# from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Optional, Type, Union, List, Dict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
import torch.nn as nn
import torch.optim as op
# import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Network import BasicBlock, BottleneckBlock, ResNet
import ipdb

def resnet_18(pretrain: bool = False) -> ResNet:
    """
    Get ResNet18
    :param pretrain: whether use pretrained model
    :return: ResNet18
    """
    return ResNet(architecture='resnet18', block=BasicBlock, layers=[2, 2, 2, 2], pretrain=pretrain)


def resnet_50(pretrain: bool = False) -> ResNet:
    """
    Get ResNet50
    :param pretrain: whether use pretrained model
    :return: ResNet50
    """
    return ResNet(architecture='resnet50', block=BottleneckBlock, layers=[3, 4, 6, 3], pretrain=pretrain)


def save_object(obj, name: str) -> None:
    """
    Save object
    :param obj: object to be saved
    :param name: name of the file
    :return: None
    """
    if not os.path.exists('./model'):
        os.mkdir('./model')
    with open(f'./model/{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(name: str):
    """
    Load object
    :param name: name of the file
    :return: the stored object
    """
    with open(f'./model/{name}.pkl', 'rb') as f:
        return pickle.load(f)

def show_results(target_model: str,
                 epochs: int,
                 accuracy: Dict[str, dict],
                 prediction: Dict[str, np.ndarray],
                 ground_truth: np.ndarray,
                 keys: List[str],
                 show_only: int) -> None:
    """
    Show accuracy results
    :param target_model: ResNet18 or ResNet50
    :param epochs: number of epochs
    :param accuracy: training and testing accuracy of different ResNets
    :param prediction: predictions of different ResNets
    :param ground_truth: ground truth of testing data
    :param keys: names of ResNet w/ or w/o pretraining
    :param show_only: Whether only show the results
    :return: None
    """
    # Get the number of characters of the longest ResNet name
    longest = len(max(keys, key=len)) + 6

    if not os.path.exists('./results'):
        os.mkdir('./results')

    # Plot
    plt.figure(0)
    plt.title(f'Result Comparison ({target_model})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    # accuracy[train][ResNet18 (w/o pretraining)][epochs] -> Accuracy of every epoch
    for train_or_test, acc in accuracy.items():
        for model in keys:
            plt.plot(range(epochs), acc[model], label=f'{model}_{train_or_test}')
            spaces = ''.join([' ' for _ in range(longest - len(f'{model}_{train_or_test}'))])
            print(f'{model}_{train_or_test}: {spaces}{max(acc[model]):.2f} %')

    plt.legend(loc='lower right')
    plt.tight_layout()
    if not show_only:
        plt.savefig(f'./results/{target_model}_comparison.png')
        plt.close()

    # prediction[ResNet18 (w/o pretraining)][ndarray] -> Best pred_labels
    for key, pred_labels in prediction.items():
        cm = confusion_matrix(y_true=ground_truth, y_pred=pred_labels, normalize='true')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
        plt.title(f'Normalized confusion matrix ({key})')
        plt.tight_layout()
        if not show_only:
            plt.savefig(f'./results/{key.replace(" ", "_").replace("/", "_")}_confusion.png')
            plt.close()

    if show_only:
        plt.show()

def train(target_model: str,
          comparison: int,
          pretrain: int,
          load_or_not: int,
          show_only: int,
          batch_size: int,
          learning_rate: float,
          epochs: int,
          optimizer: op,
          momentum: float,
          weight_decay: float,
          train_device: device,
          train_dataset: RetinopathyLoader,
          test_dataset: RetinopathyLoader) -> None:
    """
    Train the models
    :param target_model: ResNet18 or ResNet50
    :param comparison: Whether compare w/ pretraining and w/o pretraining models
    :param pretrain: Whether use pretrained model when comparison is false
    :param load_or_not: Whether load the stored model and accuracies
    :param show_only: Whether only show the results
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param optimizer: optimizer
    :param momentum: momentum for SGD
    :param weight_decay: weight decay factor
    :param train_device: training device (cpu or gpu)
    :param train_dataset: training dataset
    :param test_dataset: testing dataset
    :return: None
    """
    # Setup models w/ or w/o pretraining
    info_log('Setup models ...')
    if target_model == 'ResNet18':
        if comparison:
            keys = [
                'ResNet18 (w/o pretraining)',
                'ResNet18 (w/ pretraining)'
            ]
            models = {
                keys[0]: resnet_18(),
                keys[1]: resnet_18(pretrain=True)
            }
        else:
            if pretrain:
                keys = ['ResNet18 (w/ pretraining)']
                models = {keys[0]: resnet_18(pretrain=True)}
            else:
                keys = ['ResNet18 (w/o pretraining)']
                models = {keys[0]: resnet_18()}
            if load_or_not:
                checkpoint = load(f'./model/{target_model}.pt')
                models[keys[0]].load_state_dict(checkpoint['model_state_dict'])
    else:
        if comparison:
            keys = [
                'ResNet50 (w/o pretraining)',
                'ResNet50 (w/ pretraining)'
            ]
            models = {
                keys[0]: resnet_50(),
                keys[1]: resnet_50(pretrain=True)
            }
        else:
            if pretrain:
                keys = ['ResNet50 (w/ pretraining)']
                models = {keys[0]: resnet_50(pretrain=True)}
            else:
                keys = ['ResNet50 (w/o pretraining)']
                models = {keys[0]: resnet_50()}
            if load_or_not:
                checkpoint = load(f'./model/{target_model}.pt')
                models[keys[0]].load_state_dict(checkpoint['model_state_dict'])

    # Setup accuracy structure
    info_log('Setup accuracy structure ...')
    if show_only:
        accuracy = load_object(name='accuracy')
    elif not comparison and load_or_not:
        last_accuracy = load_object(name='accuracy')
        accuracy = {
            'train': {key: last_accuracy['train'][key] + [0 for _ in range(epochs)] for key in keys},
            'test': {key: last_accuracy['test'][key] + [0 for _ in range(epochs)] for key in keys}
        }
    else:
        accuracy = {
            'train': {key: [0 for _ in range(epochs)] for key in keys},
            'test': {key: [0 for _ in range(epochs)] for key in keys}
        }

    # Setup prediction structure
    info_log('Setup prediction structure ...')
    prediction = load_object('prediction') if not comparison and load_or_not else {key: None for key in keys}

    # Load data
    info_log('Load data ...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ground_truth = np.array([], dtype=int)
    for _, label in test_loader:
        ground_truth = np.concatenate((ground_truth, label.long().view(-1).numpy()))

    # For storing model
    stored_check_point = {
        'epoch': None,
        'model_state_dict': None,
        'optimizer_state_dict': None
    }

    # Start training
    last_epoch = checkpoint['epoch'] if not comparison and load_or_not else 0
    if not show_only:
        info_log('Start training')
        for key, model in models.items():
            info_log(f'Training {key} ...')
            if optimizer is op.SGD:
                model_optimizer = optimizer(model.parameters(), lr=learning_rate, momentum=momentum,
                                            weight_decay=weight_decay)
            else:
                model_optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            if not comparison and load_or_not:
                model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            max_test_acc = 0
            model.to(train_device)
            for epoch in tqdm(range(last_epoch, epochs + last_epoch)):
                # Train model
                model.train()
                for data, label in train_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long().view(-1)

                    pred_labels = model.forward(inputs=inputs)

                    model_optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(pred_labels, labels)
                    loss.backward()
                    model_optimizer.step()

                    accuracy['train'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
                accuracy['train'][key][epoch] = 100.0 * accuracy['train'][key][epoch] / len(train_dataset)

                # Test model
                model.eval()
                with no_grad():
                    pred_labels = np.array([], dtype=int)
                    for data, label in test_loader:
                        inputs = data.to(train_device)
                        labels = label.to(train_device).long().view(-1)

                        outputs = model.forward(inputs=inputs)
                        outputs = tensor_max(outputs, 1)[1]
                        pred_labels = np.concatenate((pred_labels, outputs.cpu().numpy()))

                        accuracy['test'][key][epoch] += (outputs == labels).sum().item()
                    accuracy['test'][key][epoch] = 100.0 * accuracy['test'][key][epoch] / len(test_dataset)

                    if accuracy['test'][key][epoch] > max_test_acc:
                        max_test_acc = accuracy['test'][key][epoch]
                        prediction[key] = pred_labels

                debug_log(f'Train accuracy: {accuracy["train"][key][epoch]:.2f}%')
                debug_log(f'Test accuracy: {accuracy["test"][key][epoch]:.2f}%')
            print()
            # Delete line286 for save w/ pretrainde and w/o pretrained model at the same time.
            # But we need to change model name with pretrained or not pretrained.
            if not comparison:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                stored_check_point['epoch'] = last_epoch + epochs
                stored_check_point['model_state_dict'] = model.state_dict()
                stored_check_point['optimizer_state_dict'] = model_optimizer.state_dict()
                save(stored_check_point, f'./model/{target_model}.pt')
                save_object(obj=accuracy, name='accuracy')
                save_object(obj=prediction, name='prediction')
            cuda.empty_cache()

    # Show results
    show_results(target_model=target_model,
                 epochs=last_epoch + epochs if not show_only else last_epoch,
                 accuracy=accuracy,
                 prediction=prediction,
                 ground_truth=ground_truth,
                 keys=keys, show_only=show_only)

def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()

def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    target_model = 'ResNet18'  # ResNet18 or ResNet50
    comparison = 0 # 'Whether compare the accuracies of w/ pretraining and w/o pretraining models'
    pretrain = 1 # Train w/ pretraining model or w/o pretraining model when "comparison" is false
    load_or_not = 0 # Whether load the stored model and accuracies
    show_only = 0
    batch_size = 4
    learning_rate = 1e-3
    epochs = 10
    optimizer = op.SGD # SGD, Adam, AdamW, Adadelta, Adamax, Adagrad
    momentum = 0.9 # Momentum factor for SGD
    weight_decay = 5e-4 # Weight decay (L2 penalty)
    global verbosity
    verbosity = 0 # 0, 1, 2

    # Read data
    info_log('Reading data ...')
    train_dataset = RetinopathyLoader('./data', 'train')
    test_dataset = RetinopathyLoader('./data', 'test')

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}')

    # Train models
    train(target_model=target_model,
          comparison=comparison,
          pretrain=pretrain,
          load_or_not=load_or_not,
          show_only=show_only,
          batch_size=batch_size,
          learning_rate=learning_rate,
          epochs=epochs,
          optimizer=optimizer,
          momentum=momentum,
          weight_decay=weight_decay,
          train_device=train_device,
          train_dataset=train_dataset,
          test_dataset=test_dataset)

if __name__ == '__main__':
    verbosity = None
    main()