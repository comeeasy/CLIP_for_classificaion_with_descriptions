import torch
import torchvision

from torch import optim, nn
from models.common import CLIPBaseModel



class CLIP_ViT_base_patch32(CLIPBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.clip, self.processor = self.get_clip_model_preprocessor("openai/clip-vit-base-patch32")
        self.fc_layer = self.add_fc_layer()
        
        # freeze modules lock or unlock comments if you want 
        self.freeze_module(self.clip.text_model)
        
        # save hyperparameters to hparams.yaml
        self.save_hyperparameters()

    def add_fc_layer(self):
        fc_layer = nn.Sequential(
            nn.Linear(768, 512, bias=True),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout1d(),
            nn.Linear(512, self.num_classes, bias=True),
        )
        self.initialize_linear_layer(fc_layer)
        
        return fc_layer
    
    def configure_optimizers(self):
        optimizer = optim.AdamW( \
            list(self.clip.vision_model.parameters()) + \
            list(self.clip.visual_projection.parameters()) + \
            list(self.fc_layer.parameters()),
            
            lr=self.hparams.lr, betas=self.hparams.optim_betas, 
            eps=self.hparams.optim_eps, weight_decay=self.hparams.optim_weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        return [optimizer], [lr_scheduler]
    

class CLIP_ViT_large_patch14(CLIPBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.clip, self.processor = self.get_clip_model_preprocessor("openai/clip-vit-large-patch14")
        self.fc_layer = self.add_fc_layer()
        
        # freeze modules lock or unlock comments if you want 
        self.freeze_module(self.clip.text_model)
        
        # save hyperparameters to hparams.yaml
        self.save_hyperparameters()

    def add_fc_layer(self):
        fc_layer = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, self.num_classes, bias=True),
        )
        self.initialize_linear_layer(fc_layer)
        
        return fc_layer
    
    def configure_optimizers(self):
        optimizer = optim.AdamW( \
            list(self.clip.vision_model.parameters()) + \
            list(self.clip.visual_projection.parameters()) + \
            list(self.fc_layer.parameters()),
            
            lr=self.hparams.lr, betas=self.hparams.optim_betas, 
            eps=self.hparams.optim_eps, weight_decay=self.hparams.optim_weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        return [optimizer], [lr_scheduler]
    

# Custeom model example
class CLIP_ConvNeXt_large_patch14(CLIPBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.clip, self.processor = self.get_clip_model_preprocessor("openai/clip-vit-large-patch14")
        self.__convert_clip_vit_to_convnext() # convert clip.vision_model to convnext
        self.fc_layer = self.add_fc_layer()
        
        # freeze modules lock or unlock comments if you want 
        self.freeze_module(self.clip.text_model)
        self.freeze_module(self.clip.text_projection)
        
        # save hyperparameters to hparams.yaml
        self.save_hyperparameters()
    
    def forward(self, imgs):
        inputs = self.processor(text=self.descs, images=imgs, return_tensors='pt', padding=True)
        inputs = inputs.to('cuda')
        
        I_f = self.clip.vision_model(imgs) # we are not using processor's output
        T_f = self.clip.text_model(inputs["input_ids"])
        
        logits_fc = self.fc_layer(I_f)
        # # joint multimodal embedding [n, d_e]
        I_e = self.clip.visual_projection(I_f)
        T_e = self.clip.text_projection(T_f['pooler_output'])
    
        # scaled pairwise cosine similarities [n, n]
        logits = self.hparams['temperature'] * torch.matmul(I_e,  T_e.T)
        
        outputs = {"text_embeds": T_e, "image_embeds": I_e}
        return logits, logits_fc, outputs
    
    def add_fc_layer(self):
        fc_layer = nn.Sequential(
            nn.LayerNorm((1536, 1, 1), eps=1e-06, elementwise_affine=False),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1536, 1024, bias=False),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, self.num_classes, bias=True),
        )
        self.initialize_linear_layer(fc_layer)
        self.initialize_linear_layer(self.clip.visual_projection)
        
        return fc_layer
    
    def configure_optimizers(self):
        optimizer = optim.AdamW( \
            list(self.clip.vision_model.parameters()) + \
            list(self.clip.visual_projection.parameters()) + \
            list(self.fc_layer.parameters()),
            
            lr=self.hparams.lr, betas=self.hparams.optim_betas, 
            eps=self.hparams.optim_eps, weight_decay=self.hparams.optim_weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        return [optimizer], [lr_scheduler]
    
    def __convert_clip_vit_to_convnext(self):
        convnext = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)
        convnext = nn.Sequential(
            *list(convnext.children())[:-1]
        )
        
        self.clip.vision_model = convnext
        self.clip.visual_projection = nn.Sequential(
            nn.LayerNorm((1536, 1, 1), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1536, 1024, bias=False),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, 768, bias=True) # T_e.shape: [batch size, 768]
        )

class CLIP_ConvNeXt_base_patch32(CLIPBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.clip, self.processor = self.get_clip_model_preprocessor("openai/clip-vit-base-patch32")
        self.__convert_clip_vit_to_convnext() # convert clip.vision_model to convnext
        self.fc_layer = self.add_fc_layer()
        
        # freeze modules lock or unlock comments if you want 
        self.freeze_module(self.clip.text_model)
        self.freeze_module(self.clip.text_projection)
        
        # save hyperparameters to hparams.yaml
        self.save_hyperparameters()
    
    def forward(self, imgs):
        inputs = self.processor(text=self.descs, images=imgs, return_tensors='pt', padding=True)
        inputs = inputs.to('cuda')
        
        I_f = self.clip.vision_model(imgs) # we are not using processor's output
        T_f = self.clip.text_model(inputs["input_ids"])
        
        logits_fc = self.fc_layer(I_f)
        # # joint multimodal embedding [n, d_e]
        I_e = self.clip.visual_projection(I_f)
        T_e = self.clip.text_projection(T_f['pooler_output'])
    
        # scaled pairwise cosine similarities [n, n]
        logits = self.hparams['temperature'] * torch.matmul(I_e,  T_e.T)
        
        outputs = {"text_embeds": T_e, "image_embeds": I_e}
        return logits, logits_fc, outputs
    
    def add_fc_layer(self):
        fc_layer = nn.Sequential(
            nn.LayerNorm((1536, 1, 1), eps=1e-06, elementwise_affine=False),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1536, 1024, bias=False),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, self.num_classes, bias=True),
        )
        self.initialize_linear_layer(fc_layer)
        self.initialize_linear_layer(self.clip.visual_projection)
        
        return fc_layer
    
    def configure_optimizers(self):
        optimizer = optim.AdamW( \
            list(self.clip.vision_model.parameters()) + \
            list(self.clip.visual_projection.parameters()) + \
            list(self.fc_layer.parameters()),
            
            lr=self.hparams.lr, betas=self.hparams.optim_betas, 
            eps=self.hparams.optim_eps, weight_decay=self.hparams.optim_weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        return [optimizer], [lr_scheduler]
    
    def __convert_clip_vit_to_convnext(self):
        convnext = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)
        convnext = nn.Sequential(
            *list(convnext.children())[:-1]
        )
        
        self.clip.vision_model = convnext
        self.clip.visual_projection = nn.Sequential(
            nn.LayerNorm((1536, 1, 1), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1536, 1024, bias=False),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, 512, bias=True)
        )
        
class CLIP_ResNet50_large_patch14(CLIPBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.clip, self.processor = self.get_clip_model_preprocessor("openai/clip-vit-large-patch14")
        
        # If you want to replace clip's vit to resnet, You must modify forward method -
        # following the method of this class, CLIP_ResNet50_large_patch14
        self.replace_clip_vit_to_resnet()   # replace ViT to ResNet50 
        self.fc_layer = self.add_fc_layer()
        
        # freeze modules lock or unlock comments if you want 
        self.freeze_module(self.clip.text_model)
        
        # save hyperparameters to hparams.yaml
        self.save_hyperparameters()

    def forward(self, imgs):
        inputs = self.processor(text=self.descs, images=imgs, return_tensors='pt', padding=True)
        inputs = inputs.to('cuda')
        
        print(imgs[0].mean(), imgs[0].std())
        
        I_f = self.clip.vision_model(inputs["pixel_values"]) # we are not using processor's output
        T_f = self.clip.text_model(inputs["input_ids"])
        
        logits_fc = self.fc_layer(I_f) # ViT uses I_f['pooler_output'] but not custom cnn models
        # # joint multimodal embedding [n, d_e]
        I_e = self.clip.visual_projection(I_f) # ViT uses I_f['pooler_output'] but not custom cnn models
        T_e = self.clip.text_projection(T_f['pooler_output'])
    
        # scaled pairwise cosine similarities [n, n]
        logits = self.hparams['temperature'] * torch.matmul(I_e,  T_e.T)
        
        outputs = {"text_embeds": T_e, "image_embeds": I_e}
        return logits, logits_fc, outputs

    def add_fc_layer(self):
        fc_layer = nn.Sequential(
            nn.Flatten(), # I_f.shape: [b, 2048, 1, 1]
            nn.Linear(2048, 1024, bias=True),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, self.num_classes, bias=True),
        )
        self.initialize_linear_layer(fc_layer)
        
        return fc_layer
    
    def configure_optimizers(self):
        optimizer = optim.AdamW( \
            list(self.clip.vision_model.parameters()) + \
            list(self.clip.visual_projection.parameters()) + \
            list(self.fc_layer.parameters()),
            
            lr=self.hparams.lr, betas=self.hparams.optim_betas, 
            eps=self.hparams.optim_eps, weight_decay=self.hparams.optim_weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        return [optimizer], [lr_scheduler]
    
    def replace_clip_vit_to_resnet(self):
        cnn_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        cnn_model = nn.Sequential(
            *list(cnn_model.children())[:-1]
        )
        
        self.clip.vision_model = cnn_model
        self.clip.visual_projection = nn.Sequential(
            nn.Flatten(), # I_f.shape: [b, 2048, 1, 1]
            nn.Linear(2048, 1024, bias=True), 
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout1d(),
            nn.Linear(1024, 768, bias=True) # T_e.shape: [batch size, 768]
        )