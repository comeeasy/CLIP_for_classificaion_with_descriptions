import os
os.environ["TOKENIZERS_PARALLELISM"] = "1"

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from src.data_module import DataModuleForCLIP
from src.utils import check_preparations, get_num_classes
from models.clips import *

from cfg import CFG



if __name__ == "__main__":
    check_preparations()
    
    CFG.num_classes = get_num_classes()
    
    #########################################
    # Select model
    #########################################
    # model = CLIP_ViT_base_patch32()
    # model = CLIP_ViT_large_patch14()
    model = CLIP_ConvNeXt_large_patch14()
    # model = CLIP_ConvNeXt_base_patch32()
    # model = CLIP_ResNet50_large_patch14()
    #########################################
    
    data_module = DataModuleForCLIP(
        train_dset_path=CFG.train_dataset_dir,
        test_size=CFG.test_size,
        batch_size=CFG.batch_size,
        width=CFG.img_transform_size_W,
        height=CFG.img_transform_size_H,
        train_transforms=CFG.train_transforms,
        val_transforms=CFG.val_transforms,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_F1Score',
        mode="max",
        filename='wallpaper_fault-{epoch:02d}-{val_F1Score:.2f}',
        save_last=True,
        every_n_epochs=100,
    )
    
    trainer = pl.Trainer(
        max_epochs=200, 
        devices="auto", 
        callbacks=[checkpoint_callback], 
        strategy=DDPStrategy(find_unused_parameters=True), 
        benchmark=True
    )
    trainer.fit(model=model, datamodule=data_module)

    trainer.test(model=model, datamodule=data_module)