# CLIP for classification using description.
----
# Introduction

CLIP is trained with huge of (image, caption) paired data. Therefore we can assume that the model, CLIP, has its “general” knowledge between image and text captured from nature. 

Consequently, CLIP is the most beloved model for zero-shot image or text tasks. However it is not feasible to apply for specific “**subtle**” tasks. For instance, classifying what type of wallpaper fault an image is certainly hard for CLIP as it is not “general”.

Therefore we need to fine-tune CLIP model to apply it for specific tasks. It may is done by following **[Finetune like you pretrain: Improved finetuning of zero-shot vision models](http://openaccess.thecvf.com/content/CVPR2023/html/Goyal_Finetune_Like_You_Pretrain_Improved_Finetuning_of_Zero-Shot_Vision_Models_CVPR_2023_paper.html).** 

Here is the things.

1. Implemented code to use CLIP of HuggingFace model.
2. Use Pytorch Lightning to fully utilize machine power.
3. Constrain training procedure using user-defined descriptions.

# Installation

```bash
conda env create -f environments.yaml
```

# Usage

1. Create your dataset for classification.
2. The dataset has to follow below structure.
    
    ```bash
    train_dataset_dir
    ├── class0
    ├── class1
    ├── class2
    ...
    ├── classN-3
    ├── classN-2
    └── classN-1
    ```
    
3. Open [cfg.py](http://cfg.py) and write a path of `train_dataset_dir` (e.g. /home/path/to/dset)
4. Open info/class_descriptions.yaml and Write descriptions for “***every***” each class as follow
    
    ```yaml
    "가구수정": separation of wallpaper that is attached next to furniture with compartments, a defect that occurs in places like built-in wardrobes or drawers
    "걸레받이수정": gap that has occurred between the mopboard and the wallpaper, mopboard is a material used to connect the side of the wallpaper and the floor, and this describes a defect that has occurred around that area
    "곰팡이": blue mold that occurs between the wallpaper surface or moldings and the wallpaper. The defect arises over time in damp conditions or due to water leakage
    ...
    classK: descrption for classK
    ...
    "틈새과다": excessive gaps that have occurred between the wallpaper surface and moldings
    "피스": tearing that occurred in the wallpaper surface due to improperly installed screws
    "훼손": damage to the wallpaper surface itself and the damage occurring between the wallpaper and moldings
    ```
    
5. Train!
    
    ```yaml
    python train.py
    ```
    

# See CFG in cfg.py

- It governs to every training procedure.

```python
class CFG():
	# model 
  batch_size = 1 # for now, batch size is fixed to 1
  img_transform_size_W = img_transform_size_H = 512
  num_classes = -1 # automatically calculated by train_dataset_dir
  label_smoothing = 0.1 # use label smoothing
  
  sim_weight = 0.5 # weight that multiplied to similarity loss proposed in paper CLIP.
  fc_weight = 0.5  # weight that multiplied to classificatio loss

	# optimizer setting you should 
  lr = 1e-6
  optim_betas = (0.9,0.98)
  optim_eps = 1e-8
  optim_weight_decay = 0.05
  temperature = 1.072508 # (exp(t)), t=0.07 from CLIP paper
  
  # dataset
  test_size = 0.2
  train_transforms = transforms.Compose([
                      transforms.Resize((img_transform_size_W, img_transform_size_H)),
                      transforms.TrivialAugmentWide(),
                      transforms.ToTensor(),
                  ])
  val_transforms = transforms.Compose([
                      transforms.Resize((img_transform_size_W, img_transform_size_H)),
                      transforms.ToTensor(),
                  ])
  train_dataset_dir = "/path/to/training/dataset/of/yours"
  class_description_yaml_file = "/path/to/class_description/yaml/written/in/step4"	

```

# References

[https://dacon.io/competitions/official/236082/data](https://dacon.io/competitions/official/236082/data)

[https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/](https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/)

[https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing](https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing)

[https://arxiv.org/pdf/2311.12065.pdf](https://arxiv.org/pdf/2311.12065.pdf)

[https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html](https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html)

[https://youtu.be/jzJHgL9VGV4?si=R_FQq_XerelO6tRB](https://youtu.be/jzJHgL9VGV4?si=R_FQq_XerelO6tRB)