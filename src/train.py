import argparse
import json
import pathlib
import sys

import numpy as np
from skimage import io
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyMSDtorch.core import helpers
from pyMSDtorch.core.networks import MSDNet

from seg_helpers import training
from seg_helpers.model import TrainingParameters


def read_data(path_imgs, path_masks):
    """
    Find all .tif training images and .npy training masks.
    Returns training images and masks as numpy arrays.
    """
    path_imgs = pathlib.Path(path_imgs)
    path_masks = pathlib.Path(path_masks)

    images = np.r_[[io.imread(img) for img in path_imgs.glob("*_for_training.tif")]]
    masks = [np.genfromtxt(mask) for mask in path_masks.glob('n-*')]
    masks = np.stack(masks, axis=0)

    # Check that the number of training images is equal to the number of masks
    assert (images.shape[0] == masks.shape[0])
    if len(images.shape) == 3:
        images = np.expand_dims(images, 1)
        masks = np.expand_dims(masks, 1)
    else:
        images = np.transpose(images, (0,3,1,2))    # (# images, # channels, x-size, y-size)
    print('number of training images:\t{}'.format(len(images)))

    return images, masks


def load_data(imgs, masks, shuffle=False, batch_size=32, num_workers=0, pin_memory=False):
    imgs = torch.Tensor(imgs)
    masks = torch.Tensor(masks)
    trainloader = TensorDataset(imgs, masks)

    loader_params = {'batch_size': batch_size,
                     'shuffle': shuffle,
                     'num_workers': num_workers,
                     'pin_memory': pin_memory}

    trainloader = DataLoader(trainloader, **loader_params)
    return trainloader


def build_network(num_classes, img_size, num_layers=10, max_dilation=10):
    in_channels = img_size[0]
    out_channels = num_classes
    num_layers = num_layers
    layer_width = 1
    max_dilation = max_dilation
    activation = nn.ReLU()      # no hardcode
    normalization = nn.BatchNorm2d  # no hardcode
    final_layer = nn.Softmax(dim=1)
    convolution = nn.Conv2d

    network = MSDNet.MixedScaleDenseNetwork(in_channels=in_channels,
                                            out_channels=out_channels,
                                            num_layers=num_layers,
                                            layer_width=layer_width,
                                            max_dilation=max_dilation,
                                            activation=activation,
                                            normalization=normalization,
                                            final_layer=final_layer,
                                            convolution=convolution
                                            )
    return network


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_dir', help='path to mask directory')
    parser.add_argument('feature_dir', help='path to feature directory')
    parser.add_argument('model_dir', help='path to model (output) directory')
    parser.add_argument('parameters', help='dictionary that contains training parameters')
    args = parser.parse_args()

    [train_imgs, train_masks] = read_data(args.feature_dir, args.mask_dir)
    img_size = train_imgs.shape

    # Load training parameters
    parameters = TrainingParameters(**json.loads(args.parameters))

    # Arrange label definition (when nonconsecutive)
    labels = list(np.unique(train_masks))
    num_classes = len(labels) - 1
    labels = labels[1:]      # remove non-labeled pixels
    ordered_labels = list(range(num_classes))
    tmp = np.copy(train_masks)
    if not labels == ordered_labels:
        for count, label in enumerate(labels):
            train_masks[tmp==label] = count
    else:
        labels = None
    print('number of classes:\t', num_classes, flush=True)

    # Define network parameters and define network
    net = build_network(num_classes,
                        img_size[1:],
                        num_layers=parameters.num_layers,
                        max_dilation=parameters.max_dilation)

    # Define training parameters
    label2ignore = -1
    criterion = getattr(nn, parameters.criterion.value)
    criterion = criterion(ignore_index=label2ignore,
                          size_average=None)

    learning_rate = parameters.learning_rate
    optimizer = getattr(optim, parameters.optimizer.value)
    optimizer = optimizer(net.parameters(), lr=learning_rate)

    epochs = parameters.num_epochs

    device = helpers.get_device()
    print('Device we will compute on:\t', device, flush=True)

    # Dataloader
    if parameters.load is not None:
        trainloader = load_data(train_imgs,
                                 train_masks,
                                 parameters.load.shuffle,
                                 parameters.load.batch_size,
                                 parameters.load.num_workers,
                                 parameters.load.pin_memory)
    else:
        trainloader = load_data(train_imgs,
                                 train_masks)

    # Begin Training
    net.to(device)  # send network to device
    net, train_loss = training.train_segmentation(net,
                                                  trainloader,
                                                  epochs,
                                                  criterion,
                                                  optimizer,
                                                  device)

    param_count = helpers.count_parameters(net)
    print('number of network parameters:\t{}'.format(param_count), flush=True)

    ## Save model
    model_output_name = args.model_dir + '/state_dict_net.pt'
    torch.save({'model': net, 'labels': labels}, model_output_name)
    torch.cuda.empty_cache()
