from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

class NNModel(str, Enum):
    msdnet = 'MSDNet'
    tunet = 'TUNet'
    tunet3plus = 'TUNet3+'


class Optimizer(str, Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamw = "AdamW"
    sparseadam = "SparseAdam"
    adamax = "Adamax"
    asgd = "ASGD"
    lbfgs = "LBFGS"
    rmsprop = "RMSprop"
    rprop = "Rprop"
    sgd = "SGD"


class Criterion(str, Enum):
    l1loss = "L1Loss" 
    mseloss = "MSELoss" 
    crossentropyloss = "CrossEntropyLoss"
    ctcloss = "CTCLoss"
    poissonnllloss = "PoissonNLLLoss"
    gaussiannllloss = "GaussianNLLLoss"
    kldivloss = "KLDivLoss"
    bceloss = "BCELoss"
    bcewithlogitsloss = "BCEWithLogitsLoss"
    marginrankingloss = "MarginRankingLoss"
    hingeembeddingloss = "HingeEnbeddingLoss"
    multilabelmarginloss = "MultiLabelMarginLoss"
    huberloss = "HuberLoss"
    smoothl1loss = "SmoothL1Loss"
    softmarginloss = "SoftMarginLoss"
    multilabelsoftmarginloss = "MutiLabelSoftMarginLoss"
    cosineembeddingloss = "CosineEmbeddingLoss"
    multimarginloss = "MultiMarginLoss"
    tripletmarginloss = "TripletMarginLoss"
    tripletmarginwithdistanceloss = "TripletMarginWithDistanceLoss"


class LoadParameters(BaseModel):
    shuffle: Optional[bool] = Field(description="shuffle data")
    batch_size: Optional[int] = Field(description="batch size")
    num_workers: Optional[int] = Field(description="number of workers")
    pin_memory: Optional[bool] = Field(description="memory pinning")


class MSDNetParameters(BaseModel):
    num_layers: int = Field(description="number of layers")
    custom_dilation: bool = Field(description="whether to customize dilation")
    max_dilation: Optional[int] = Field(description="maximum dilation")
    dilation_array: Optional[List[int]] = Field(description="customized dilation array")


class TUNetParameters(BaseModel):
    depth: int = Field(description='the depth of the UNet')
    base_channels: int = Field(description='the number of initial channels')
    growth_rate: int = Field(description='multiplicative growth factor of number of channels per layer of depth')
    hidden_rate: int = Field(description='multiplicative growth factor of channels within each layer')
    carryover_channels: Optional[int] = Field(description='the number of channels in each skip connection')


class TrainingParameters(BaseModel):
    model: NNModel
    num_epochs: int = Field(description="number of epochs")
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    msdnet_parameters: Optional[MSDNetParameters]
    tunet_parameters: Optional[TUNetParameters]
    load: Optional[LoadParameters]


class Metadata(BaseModel):
    epoch: int = Field(description='epoch')
    loss: float = Field(description='loss')


class TestingParameters(BaseModel):
    show_progress: int = Field(description="number of iterations to progress report")
    load: Optional[LoadParameters]


class TestingResults(BaseModel):
    img_idx: int = Field(description="index of segmented image")
