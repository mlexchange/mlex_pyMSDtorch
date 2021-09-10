import torch
from seg_helpers.model import Metadata


def train_segmentation(net,
                       trainloader,
                       num_epochs,
                       criterion,
                       optimizer,
                       device):
    """
    Loop through epochs passing noisy images to net.

    Input:
        :param net: The neural network
        :type net: Instantiated class from pyMSDtorch.core.networks
        :param trainloader: Stack of training images and images_masks
        :type trainloader: Dataloader object
        :param num_epochs: Number of training epochs
        :type num_epochs: Int
        :param criterion: User-specified loss function
            See https://pytorch.org/docs/stable/nn.html#loss-functions
            OR check out pyMSDtorch/core/loss_functions.py for custom
                binary loss functions
        :type criterion: Instantiated class, typically from
                         torch.nn.modules.loss
        :param optimizer: Pytorch optimizer
            See https://pytorch.org/docs/stable/optim.html
        :type optimizer: Instantiated class from torch.optim
        :param device: Device on which the training occurs on, typically
                       'cpu' if no GPU or 'cuda:x' where x is the GPU
                       index
        :type device: Str

    """
    train_loss = []

    # Create log file
    f = open('logs.txt', 'w')
    header = list(Metadata.__fields__)
    f.write(",".join(header) + "\n")

    for epoch in range(num_epochs):
        running_train_loss = 0.0

        for data in trainloader:  # loop through each batch

            noisy, target = data  # load noisy and target images #not transformed (zero is unmasked)

            noisy = noisy.type(torch.FloatTensor)  # Cast data as tensor of floats
            target = target.type(torch.FloatTensor)
            noisy = noisy.to(device)
            target = target.to(device)

            # For multi-class segmentation using Cross Entropy, PyTorch
            # requires the target to be a tensor of Longs squeezed along
            # channel dimensions
            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(device).squeeze(1)

            output = net(noisy)
            loss = criterion(output, target)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            # update the parameters
            optimizer.step()
            running_train_loss += loss.item()

        loss = running_train_loss / len(trainloader)
        train_loss.append(loss)

        f.write(str(epoch) + ',' + str(loss) + "\n")

    f.close

    return net, train_loss
