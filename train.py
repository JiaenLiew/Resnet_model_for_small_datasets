import argparse
import os
import tensorflow as tf
from typing import Any

from sklearn.model_selection import train_test_split
from PIL import Image

from common import load_image_labels, load_single_image, save_model

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
import math

########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args


## So far, the only rescourses we need is the model
def load_train_resources():
    """
    Load any resources (i.e. pre-trained models, data files, etc) here.
    Make sure to submit the resources required for your algorithms in the sub-folder 'resources'
    :param resource_dir: the relative directory from train.py where resources are kept.
    :return: model?
    """
    ## Model configuration and customisation
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

    return model


def train(train_dataset, val_dataset, output_dir: str, model):
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.

    # raise RuntimeError("train() is not implemented.")

    # img_len = len(images)
    # print(img_len)

    # p_img = []

    # for img in images:
    #     x =
    alpha = 1e-7
    train_bs = math.floor(len(train_dataset)/5)
    eval_bs = 1
    metric = evaluate.load("f1")
    normalizer = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    optimizer = optim.AdamW(params=model.parameters())

    train_transforms = v2.Compose([
                        Resize((224, 224)),
                        v2.RandomHorizontalFlip(), 
                        v2.RandomRotation(10),
                        v2.ColorJitter(brightness=0.2, contrast=0.2),
                        ToTensor(),
                        normalizer
                        ])

    val_transforms = v2.Compose([
                        Resize((224, 224)),
                        ToTensor(),
                        normalizer
                        ])

    def train_transform(data):
        data["pixel_values"] = [train_transforms(image) for image in data['Image']]
        return data

    def test_transform(data):
        data["pixel_values"] = [val_transforms(image) for image in data['Image']]
        return data

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        print(predictions)
        print(eval_pred.label_ids)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([example['labels'] for example in examples])
        return {'pixel_values': pixel_values, 'labels': labels}

    train_dataset.set_transform(train_transform)
    val_dataset.set_transform(test_transform)

    huggingface_resnet_18 = "microsoft/resnet-18"
    training_args = TrainingArguments(
        huggingface_resnet_18,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=alpha,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=200,
        logging_steps=10,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        remove_unused_columns=False,
        report_to='tensorboard'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)

    return None


def main(image_path: str, csv_file_path: str, labels: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """
    csv_file_path = os.path.join(image_path, csv_file_path)
    csv_data = pd.read_csv(csv_file_path)

    image_paths = []
    labels = []

    for index, row in csv_data.iterrows():
        image_path1 = os.path.join(image_path, row['Filename'])
        print(f"Image path{image_path}")
        label = row[1]
        print(label)
        
        image_paths.append(Image.open(image_path1).convert('RGB'))

        if label == 'Yes':
            labels.append(1)
        else:
            labels.append(0)

    df = pd.DataFrame({'Image': image_paths, 'labels': labels})

    dataset = Dataset.from_dict(df)

    test_size = 0.2
    if ((int(len(dataset) - (len(dataset) * 0.2)) % 2) != 0):
        test_size = 0.1

    dataset = dataset.class_encode_column('labels')
    split_dataset = dataset.train_test_split(test_size=test_size, stratify_by_column='labels')
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    model = load_train_resources()
    train(train_dataset, val_dataset, train_output_dir, model)

    return None


if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
   
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)
    print("* "*100)

########################################################################################################################