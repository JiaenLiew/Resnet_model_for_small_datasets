import argparse
import os
import tensorflow as tf
from typing import Any

from sklearn.model_selection import train_test_split
from PIL import Image

from common import load_single_image, load_model, load_predict_image_names

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
import os
from sklearn.utils import compute_class_weight
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from PIL import Image

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

import tensorflow as tf
import torchvision.models as models
from transformers.models.auto.modeling_auto import AutoModelForImageClassification
from typing import List
from timm.models.resnet import BasicBlock, Bottleneck, ResNet

########################################################################################################################
def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args


def predict(
            csv_file_path: str,
            images_path: str,
            trained_model_dir: str,) -> str:
    """
    Generate a prediction for a single image using the model, returning a label of 'Yes' or 'No'
    IMPORTANT: The return value should ONLY be either a 'Yes' or 'No' (Case sensitive)

    :param model: the model to use.
    :param image: the image file to predict.
    :return: the label ('Yes' or 'No)
    """
    
    ## Model configuration and customisation
    
    model = load_model(trained_model_dir, target_column_name)
    normalizer = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    csv_data = pd.read_csv(csv_file_path)

    image_paths = []
    labels = []

    for index, row in csv_data.iterrows():
        image_path = os.path.join(images_path, row['Filename'])
        label = row[1]
        
        image_paths.append(Image.open(image_path).convert('RGB'))

        if label == 'Yes':
            labels.append(1)
        else:
            labels.append(0)

    df = pd.DataFrame({'Image': image_paths, 'labels': labels})

    dataset = Dataset.from_dict(df)

    val_transforms = v2.Compose([
                        Resize((224, 224)),
                        ToTensor(),
                        normalizer
                        ])

    dataset.set_transform(val_transforms)
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset)

    if predictions == 1:
        predicted_label = 'Yes'
    else:
        predicted_label = 'No'

    return predicted_label


def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """

    # load pre-trained models or resources at this stage.
    model = load_model(trained_model_dir, target_column_name)

    # Load in the image list
    image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
    image_filenames = load_predict_image_names(image_list_file)

    # Iterate through the image list to generate predictions
    predictions = []
    for filename in image_filenames:
        try:
            image_path = os.path.join(predict_data_image_dir, filename)
            image = load_single_image(image_path)
            label = predict(model, image)
            predictions.append(label)
        except Exception as ex:
            print(f"Error generating prediction for {filename} due to {ex}")
            predictions.append("Error")

    df_predictions = pd.DataFrame({'Filenames': image_filenames, target_column_name: predictions})

    # Finally, write out the predictions to CSV
    df_predictions.to_csv(predicts_output_csv, index=False)


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
