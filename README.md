# pyMSDtorch
An image segmentation algorithm that implements Mixed-scale Dense Networks (MSDNets) using [pyMSDtorch](https://pymsdtorch.readthedocs.io/en/latest/). Currently not supporting M1 Mac.

## Getting started
To get started, you will need:
  - [Docker](https://docs.docker.com/get-docker/)

## Running
First, build the segMSDnet image in terminal:
```
cd mlex_pyMSDtorch
make build_docker
```
Once built, you can run the following examples:
```
make train_example
```
```
make test_example
```
These examples utilize the information stored in the folder /data. The trained model and the segmented images will be stored in /data/model and /data/output, respectively.

Alternatively, you can run the container interactively as follows:
```
make run_docker
```

While running interactively, you can perform training and testin processes using MSDnet.

### Training
To train an MSDnet within the docker container, you can run the command:
```
python3 src/train.py path/to/input_mask_image_dir path/to/input_train_image_dir path/to/output_trained_model_dir "parameters"
```

where:
  - path/to/input_mask_image_dir: directory that containes the input masks
  - path/to/input_train_image_dir: directory that containes the input training images
  - path/to/output_trained_model_dir: directory where the algorithm will store the trained model
  - "parameters": training parameters defined as follows,
 
        '{"num_epochs": 200, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "num_layers": 10, "max_dilation": 10}'

Further information on the training parameters can be found in scr/seg_helpers/model.py

### Testing
To test an MSDnet model within the docker container, you can run the command:
```
python3 src/test.py path/to/input_test_image_file path/to/input_trained_model_file path/to/output_segmented_images_dir "parameters"
```

where:
  - path/to/input_test_image_file: path to the test images, which includes the file name and extension
  - path/to/input_trained_model_file: path to the trained model, which includes the file name and extension
  - path/to/output_segmented_images_dir: directory where the algorithm will store the segmented images
  - "parameters": testing parameters defined as follows,
 
        '{"show_progress": 20}'

Further information on the training parameters can be found in scr/seg_helpers/model.py


## Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
