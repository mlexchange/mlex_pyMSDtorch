from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


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
    l1loss = "L1Loss" #*
    mseloss = "MSELoss" #*
    crossentropyloss = "CrossEntropyLoss"
    ctcloss = "CTCLoss"
#    nllloss = "NLLLoss"
    poissonnllloss = "PoissonNLLLoss"
    gaussiannllloss = "GaussianNLLLoss"
    kldivloss = "KLDivLoss"
    bceloss = "BCELoss" #*
    bcewithlogitsloss = "BCEWithLogitsLoss" #*
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


class TrainingParameters(BaseModel):
    num_epochs: int = Field(description="number of epochs")
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    num_layers: int = Field(description="number of layers")
    max_dilation: int = Field(description="maximum dilation")
    load: Optional[LoadParameters]


class Metadata(BaseModel):
    epoch: int = Field(description='epoch')
    loss: float = Field(description='loss')


class TestingParameters(BaseModel):
    show_progress: int = Field(description="number of iterations to progress report")
    load: Optional[LoadParameters]


class TestingResults(BaseModel):
    img_idx: int = Field(description="index of segmented image")
