import torchvision.transforms as transforms


class CFG():
    # model 
    batch_size = 1 # for now, batch size is fixed to 1
    img_transform_size_W = img_transform_size_H = 512
    num_classes = -1 # automatically calculated by train_dataset_dir
    label_smoothing = 0.1
    
    sim_weight = 0.5
    fc_weight = 0.5
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
    train_dataset_dir = "/home/hm086/joono/DACON/wallpaper_fault_classification/train"
    class_description_yaml_file = "./info/class_description.yaml"
    
    

