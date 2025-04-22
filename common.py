import argparse
import os
import tensorflow as tf
from typing import Any

from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd

from torch import optim
from torch.utils.data import *
from torchvision.datasets import *
from torchvision.transforms import *
from torchvision.transforms import v2
from datasets import load_dataset, load_metric, Dataset
import evaluate
from sklearn.utils import compute_class_weight
import torch.nn.functional as F

from transformers import (
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer, 
    ResNetForImageClassification, 
    AutoFeatureExtractor,
    ResNetConfig,
    AutoImageProcessor,
    create_optimizer,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig
)
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from transformers.models.auto.modeling_auto import AutoModelForImageClassification
from typing import List
from timm.models.resnet import BasicBlock, Bottleneck, ResNet

## To use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################################################################################################
# Data Loading functions
########################################################################################################################
def load_image_labels(labels_file_path: str):
    """
    Loads the labels from CSV file.

    :param labels_file_path: CSV file containing the image and labels.
    :return: Pandas DataFrame
    """
    df = pd.read_csv(labels_file_path)
    return df


def load_predict_image_names(predict_image_list_file: str) -> [str]:
    """
    Reads a text file with one image file name per line and returns a list of files
    :param predict_image_list_file: text file containing the image names
    :return list of file names:
    """
    with open(predict_image_list_file, 'r') as file:
        lines = file.readlines()
    # Remove trailing newline characters if needed
    lines = [line.rstrip('\n') for line in lines]
    return lines


def load_single_image(image_file_path: str) -> Image:
    """
    Load the image.

    NOTE: you can optionally do some initial image manipulation or transformation here.

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    # Load the image
    image = Image.open(image_file_path)

    # The following are examples on how you might manipulate the image.
    # See full documentation on Pillow (PIL): https://pillow.readthedocs.io/en/stable/

    # To make the image 50% smaller
    # Determine image dimensions
    # width, height = image.size
    # new_width = int(width * 0.50)
    # new_height = int(height * 0.50)
    # image = image.resize((new_width, new_height))

    # To crop the image
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # image = image.crop((left, upper, right, lower))

    # To view an image
    # image.show()

    # Return either the pixels as array - image_array
    # To convert to a NumPy array
    # image_array = np.asarray(image)
    # return image_array

    # or return the image
    return image


########################################################################################################################
# Model Loading and Saving Functions
########################################################################################################################

def save_model(model: Any, target: str, output_dir: str):
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation.

    Common Deep Learning Model File Formats are:

        SavedModel (TensorFlow)
        Pros: Framework-agnostic format, can be deployed in various environments. Contains a complete model representation.
        Cons: Can be somewhat larger in file size.

        HDF5 (.h5) (Keras)
        Pros: Hierarchical structure, good for storing model architecture and weights. Common in Keras.
        Cons: Primarily tied to the Keras/TensorFlow ecosystem.

        ONNX (Open Neural Network Exchange)
        Pros: Framework-agnostic format aimed at improving model portability.
        Cons: May not support all operations for every framework.

        Pickle (.pkl) (Python)
        Pros: Easy to save and load Python objects (including models).
        Cons: Less portable across languages and environments. Potential security concerns.

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param model: the model that you want to save.
    :param target: the target value - can be useful to name the model file for the target it is intended for
    :param output_dir: the output directory to same one or more model files.
    """
    trainer = Trainer(model)
    trainer.save_model(output_dir)


def load_model(trained_model_dir: str, target_column_name: str) -> Any:
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation and should mirror save_model()

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param trained_model_dir: the directory where the model file(s) are saved.
    :param target_column_name: the target value - can be useful to name the model file for the target it is intended for
    :returns: the model
    """

    huggingface_resnet_18 = "microsoft/resnet-18"
    BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

    class ResnetConfig(PretrainedConfig):
        model_type = "resnet"
        def __init__(
            self,
            block_type="bottleneck",
            layers: List[int] = [1, 1, 1, 1],
            num_classes: int = 2,
            input_channels: int = 3,
            cardinality: int = 1,
            base_width: int = 16,
            stem_width: int = 16,
            stem_type: str = "",
            avg_down: bool = False,
            **kwargs,
        ):        
            self.block_type = block_type
            self.layers = layers
            self.num_classes = num_classes
            self.input_channels = input_channels
            self.cardinality = cardinality
            self.base_width = base_width
            self.stem_width = stem_width
            self.stem_type = stem_type
            self.avg_down = avg_down
            super().__init__(**kwargs)
            
    class ResnetModelForImageClassification(PreTrainedModel):
        config_class = ResnetConfig
        def __init__(self, config):
            super().__init__(config)
            block_layer = BLOCK_MAPPING[config.block_type],
            self.model = ResNet(
                Bottleneck,
                config.layers,
                num_classes=config.num_classes,
                in_chans=config.input_channels,
                cardinality=config.cardinality,
                base_width=config.base_width,
                stem_width=config.stem_width,
                stem_type=config.stem_type,
                avg_down=config.avg_down,
            )

        def forward(self, pixel_values, labels=None):
            logits = self.model(pixel_values)
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
                return {"loss": loss, "logits": logits}
            return {"logits": logits}
        
    config = ResnetConfig()
    model = ResnetModelForImageClassification(config)

    class New_Classifier(nn.Module):
        def __init__(self):
            super(New_Classifier, self).__init__()
            self = nn.Flatten(1, -1)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 2)
            self.dropout = nn.Dropout(0.4)

        def forward(self, tensor, labels=None):
            logits = self.fc1(tensor)
            logits = torch.gelu(tensor)
            logits = self.fc2(tensor)
            if labels is not None:
                loss = nn.BCEWithLogitsLoss(logits, labels)
                return {"loss": loss, "logits": logits}
            return {"logits": logits}

    model.classifier = New_Classifier()

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    model.to('cuda')
    config = ResnetConfig().from_pretrained(trained_model_dir)
    model = ResnetModelForImageClassification(config).from_pretrained(trained_model_dir)

    return model
