import os
import pandas as pd
from glob import glob


def class_label_table(classes):
    # determine label for all descriptions (e.g. 가구수정 -> 0, 걸레받이수정 -> 1)
    cls2label, label2cls = {}, {}
    for n, cls in enumerate(classes):
        cls2label[cls] = int(n)
        label2cls[int(n)] = cls
        
    return cls2label, label2cls


def write_class_label_table(classes, file_name):
    cls2label, _ = class_label_table(classes)

    classes, labels = [], []
    for cls, label in cls2label.items():
        classes.append(cls)
        labels.append(label)

    # write to json
    df_cls_lb = pd.DataFrame({
        "class": classes,
        "label": range(len(classes))
    })
    df_cls_lb.to_json(file_name)

def write_naive_annotation_json(classes, file_name):
    train_data_path = os.path.abspath("train")

    # determine label for all descriptions (e.g. 가구수정 -> 0, 걸레받이수정 -> 1)
    cls2label, _ = class_label_table(classes)

    # create train data annotation file
    img_paths, labels, df_classes = [], [], []
    for cls in classes:
        for img_path in glob(os.path.join(train_data_path, cls, "*")):
            img_paths.append(img_path)
            labels.append(cls2label[cls])
            df_classes.append(cls)

    # creattion dataframe
    df = pd.DataFrame({
        "img_path": img_paths,
        "labels": labels,
        "class": df_classes 
        })

    # write to json 
    df.to_json(file_name)
    
def write_oversample_annotation_json(classes, file_name):
    train_data_path = os.path.abspath("train")

    # determine label for all descriptions (e.g. 가구수정 -> 0, 걸레받이수정 -> 1)
    cls2label, _ = class_label_table(classes)

    # create train data annotation file
    img_paths, labels, df_classes = [], [], []
    for cls in classes:
        for img_path in glob(os.path.join(train_data_path, cls, "*")):
            img_paths.append(img_path)
            labels.append(cls2label[cls])
            df_classes.append(cls)

    # creattion dataframe
    df = pd.DataFrame({
        "img_path": img_paths,
        "labels": labels,
        "class": df_classes 
        })

    # write to json 
    df.to_json(file_name)