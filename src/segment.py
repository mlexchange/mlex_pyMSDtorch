import argparse
import json

import numpy as np
import imageio
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

from dlsia.core import helpers
from model import TestingParameters, TestingResults


class PredictDataset(Dataset):
    def __init__(self, image_path):
        # assume a tiff stack for now
        self.image_path = image_path
        images = imageio.volread(image_path)
        if len(images.shape) == 3:
            images = np.expand_dims(images, 1)
        else:
            images = np.transpose(images, (0, 3, 1, 2))  # (# images, # channels, x-size, y-size)
        self.im_stack = torch.from_numpy(images)

    def __len__(self):
        return len(self.im_stack)

    def __getitem__(self, idx):
        return self.im_stack[idx].unsqueeze(0)


def load_data(img_path, shuffle=False, batch_size=1, num_workers=0, pin_memory=False):
    # create custom dataset that doesn't require masks
    predict_data = PredictDataset(img_path)

    loader_params = {'batch_size': batch_size,
                     'shuffle': shuffle,
                     'num_workers': num_workers,
                     'pin_memory': pin_memory}

    testloader = DataLoader(predict_data, **loader_params)
    return testloader


if __name__ == '__main__':
    ### GLOBAL VARIABLES ###
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_f', help='model filepath')
    parser.add_argument('output_dir', help='directory for output of classifier')
    parser.add_argument('parameters', help='dictionary that contains training parameters')
    args = parser.parse_args()

    # Load training parameters
    parameters = TestingParameters(**json.loads(args.parameters))
    if parameters.load is not None:
        testloader = load_data(args.image_stack,
                               parameters.load.shuffle,
                               parameters.load.batch_size,
                               parameters.load.num_workers,
                               parameters.load.pin_memory)
    else:
        testloader = load_data(args.image_stack)

    # load network from file
    model_dict = torch.load(args.model_f)
    net = model_dict['model']
    labels = model_dict['labels']
    net.train()     # Set network to training mode (eval mode kills batchnorm params)

    # prepare output directory
    segpath = args.output_dir
    helpers.make_dir(segpath)

    device = helpers.get_device()
    counter = 0
    for counter, batch in enumerate(testloader):
        noisy = np.squeeze(batch, axis=0)
        noisy = noisy.type(torch.FloatTensor)
        noisy = noisy.to(device)
        output = net(noisy)     # segments
        show_me = torch.argmax(output.cpu().data, dim=1)
        tmp = torch.clone(show_me)
        if labels is not None:
            for count, label in enumerate(labels):
                show_me[tmp==count] = label
        # save current image
        if counter % parameters.show_progress == 0:
            progress = show_me.cpu().detach().numpy()
            progress = (progress*255).astype(np.uint8)
            tifffile.imwrite(str(segpath) + '/{}-classified.tif'.format(counter), progress)
            #imageio.mimsave(str(segpath) + '/{}-classified.tif'.format(counter), progress)
            print('classified\t'+ str(counter), flush=True)
        if counter == 0:
            segmented_imgs = show_me
        else:
            segmented_imgs = torch.cat((segmented_imgs, show_me), 0)
    # save results
    segmented_imgs = segmented_imgs.cpu().detach().numpy()
    np.save(str(segpath) + '/results.npy', segmented_imgs)
    segmented_imgs = (segmented_imgs*255).astype(np.uint8)
    tifffile.imwrite(str(segpath) + '/results.tif', segmented_imgs)
    #imageio.mimwrite(str(segpath) + '/results.tif', segmented_imgs)
