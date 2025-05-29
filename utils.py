import torch
from torchvision import transforms
from PIL import Image


def load_image(image_path, size=256, center_crop=True):
    image = Image.open(image_path).convert("RGB")
    transform_list = []
    if size:
        transform_list.append(transforms.Resize(size))
    if center_crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0) 


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def save_image(tensor, filename):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    tensor = denormalize(tensor)
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(filename)
