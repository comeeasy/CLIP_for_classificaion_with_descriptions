import numpy as np

import lightning.pytorch as pl

import torch
from torch import optim, nn
from torchmetrics.classification import Accuracy, F1Score

from src.utils import read_class_description
from transformers import CLIPProcessor, CLIPModel

from cfg import CFG



class CLIPBaseModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = CFG.num_classes
        self.label_smoothing = CFG.label_smoothing
        
        # Loss
        self.cls_loss_i = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.cls_loss_t = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.cls_fc_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.class_descs = read_class_description()
        self.classes_list = list(self.class_descs.keys())
        self.descs = list(self.class_descs.values())
        
        # hyper parameters
        self.hparams['num_classes'] = self.num_classes
        self.hparams['label_smoothing'] = self.label_smoothing
        
        # sim loss, fc loss coefficients
        self.hparams['sim_weight'] = CFG.sim_weight
        self.hparams['fc_weight'] = CFG.fc_weight
        
        # hyperparameters
        self.hparams["lr"] = CFG.lr
        self.hparams["optim_betas"] = CFG.optim_betas
        self.hparams["optim_eps"] = CFG.optim_eps
        self.hparams["optim_weight_decay"] = CFG.optim_weight_decay
        
        self.hparams['temperature'] = CFG.temperature
        self.hparams["class_label"] = {cls: label for label, cls in enumerate(self.class_descs)}
        
        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average='weighted')
    
    def forward(self, imgs):
        # should be modified to use only text processor
        inputs = self.processor(text=self.descs, images=imgs, return_tensors='pt', padding=True)
        inputs = inputs.to('cuda')
        
        I_f = self.clip.vision_model(imgs) # Since imgs are preprocesses, we don't use inputs['pixel_values']
        T_f = self.clip.text_model(inputs["input_ids"])
        
        logits_fc = self.fc_layer(I_f['pooler_output'])
        # joint multimodal embedding [n, d_e]
        I_e = self.clip.visual_projection(I_f['pooler_output'])
        T_e = self.clip.text_projection(T_f['pooler_output'])
    
        # L2 Normalization for each embeddings
        I_e = nn.functional.normalize(I_e, p=2)
        T_e = nn.functional.normalize(T_e, p=2)
    
        # scaled pairwise cosine similarities [n, n]
        logits = self.hparams['temperature'] * torch.matmul(I_e,  T_e.T)
        
        outputs = {"text_embeds": T_e, "image_embeds": I_e}
        return logits, logits_fc, outputs
    
    def training_step(self, batch, batch_idx):
        assert batch.shape[0] == 1 # for now, batch-size is limited to 1.
        
        imgs = batch[0]
        logits, logits_fc, outputs = self(imgs)
        
        sim_loss = self.calculate_symmetric_loss(logits=logits)
        fc_loss = self.calculate_fc_loss(logits_fc=logits_fc)
        loss = self.calculate_total_loss(sim_loss=sim_loss, fc_loss=fc_loss)
        
        self.logging_embeddings_and_logits_to_tensorboard(logits=logits, outputs=outputs, batch_idx=batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        assert batch.shape[0] == 1 # for now, batch-size is limited to 1.
        
        imgs = batch[0]
        logits, logits_fc, outputs = self(imgs)
        
        sim_loss = self.calculate_symmetric_loss(logits=logits)
        fc_loss = self.calculate_fc_loss(logits_fc=logits_fc)
        loss = self.calculate_total_loss(sim_loss=sim_loss, fc_loss=fc_loss)
        
        fc_target = torch.arange(len(self.descs)).to('cuda')
        val_acc = self.valid_acc(logits_fc, fc_target)
        val_f1 = self.valid_f1(logits_fc, fc_target)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_F1Score", val_f1, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", val_acc, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        assert batch.shape[0] == 1 # for now, batch-size is limited to 1.
        
        imgs = batch[0]
        logits, logits_fc, outputs = self(imgs)
        
        sim_loss = self.calculate_symmetric_loss(logits=logits)
        fc_loss = self.calculate_fc_loss(logits_fc=logits_fc)
        loss = self.calculate_total_loss(sim_loss=sim_loss, fc_loss=fc_loss)
        
        self.log("test_loss", loss, sync_dist=True)
    
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def get_clip_model_preprocessor(self, url):
        clip = CLIPModel.from_pretrained(url)
        processor = CLIPProcessor.from_pretrained(url)
        return clip, processor
    
    def calculate_symmetric_loss(self, logits):
        assert logits.shape[0] == len(self.descs) # logits shapes must be same as [class num X class num]
        
        # symmetric loss function
        contrastive_target = torch.arange(len(self.descs)).to('cuda')
        
        loss_i = self.cls_loss_i(logits, contrastive_target)
        loss_t = self.cls_loss_t(logits.T, contrastive_target)
        loss = (loss_i + loss_t) / 2
        return loss
    
    def calculate_fc_loss(self, logits_fc):
        fc_target = torch.arange(len(self.descs)).to('cuda')
        fc_loss = self.cls_fc_loss(logits_fc, fc_target)
        return fc_loss

    def calculate_total_loss(self, sim_loss, fc_loss):
        loss = self.hparams['sim_weight'] * sim_loss + self.hparams['fc_weight'] * fc_loss
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()
        self.valid_acc.reset()
        self.valid_f1.reset()
        
    def logging_embeddings_and_logits_to_tensorboard(self, logits, outputs, batch_idx):
        T_e, I_e = outputs['text_embeds'], outputs['image_embeds']
        
        tensorboard = self.logger.experiment
        T_e_canvas = np.zeros((1, T_e.shape[0], 50))
        for i, T_e_row in enumerate(T_e):
            for j, value in enumerate(T_e_row[:50]):
                T_e_canvas[0, i, j] = value.item()
        I_e_canvas = np.zeros((1, I_e.shape[0], 50))
        for i, I_e_row in enumerate(I_e):
            for j, value in enumerate(I_e_row[:50]):
                I_e_canvas[0, i, j] = value.item()
        logits_canvas = np.zeros((1, logits.shape[0], logits.shape[1]))
        for i, logits_row in enumerate(logits):
            for j, value in enumerate(logits_row):
                logits_canvas[0, i, j] = value.item()
                
        tensorboard.add_image("text embeddings", T_e_canvas, batch_idx)
        tensorboard.add_image("image embeddings", I_e_canvas, batch_idx)
        tensorboard.add_image("logits", logits_canvas, batch_idx)
    
    def configure_optimizers(self):
        raise NotImplementedError("implement configure_optimizer. See models/clip.py!")
    
    def add_fc_layer(self):
        raise NotImplementedError("implement add_fc_layer!")
    
    def initialize_linear_layer(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)