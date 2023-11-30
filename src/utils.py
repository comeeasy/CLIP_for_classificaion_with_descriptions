import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torchvision.io import read_image as _read_image
from PIL import Image
import yaml

from glob import glob

from cfg import CFG



def check_preparations():
    class_descs = read_class_description()
    
    if len(glob(os.path.join(CFG.train_dataset_dir, "*"))) != len(class_descs.keys()):
        print(f"train classes in dataset: {len(glob(os.path.join(CFG.train_dataset_dir, '*')))}, classes in class_description.yaml: {len(class_descs.keys())}")
        raise RuntimeError(f"You have to write every descriptions for each classes in {CFG.train_dataset_dir} to class_descriptions.yaml")
    
    print(f"Here is the table of class-description paires: \n")
    for cls, desc in class_descs.items():
        print(f"[{cls}]: {desc}")
    print(f"=" * 30)
    
def get_num_classes():
    return len(glob(os.path.join(CFG.train_dataset_dir, "*")))

def read_image(img_path):
    
    return Image.open(img_path)
    # return _read_image(img_path, torchvision.io.ImageReadMode.RGB).float()

def draw_confusion_matrix(cf_mat):
    import seaborn as sns

    ax = sns.heatmap(cf_mat, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
def read_class_description() -> dict:
    with open(CFG.class_description_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data

def read_class_label() -> dict:
    with open(CFG.class_label_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def read_hparams(hparam_path) -> dict:
    with open(hparam_path, "r") as f:
        data = yaml.full_load(f)
    return data

if __name__ == "__main__":
    print("hello world")