import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import load_image, save_image, denormalize
from model import TransformerNet

image_size = 256
batch_size = 4
epochs = 600
style_weight = 70
learning_rate = 1e-3
style_image_path = "images/style.jpg"
dataset_path = "resized_output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
dataset = ImageFolder(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

style_image = load_image(style_image_path, size=image_size).to(device)

transformer = TransformerNet().to(device)
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:23].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

mse_loss = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

style_features = []
with torch.no_grad():
    x = style_image
    for layer in vgg:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            style_features.append(x)
    style_grams = [gram_matrix(f) for f in style_features]

os.makedirs("stylized_training_output", exist_ok=True)
print("Training started...")
step = 0
for epoch in range(epochs):
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        y_hat = transformer(x)

        y_features, x_features = [], []
        fx, fy = x, y_hat
        for layer in vgg:
            fx = layer(fx)
            fy = layer(fy)
            if isinstance(layer, nn.ReLU):
                x_features.append(fx)
                y_features.append(fy)

        content_loss = mse_loss(y_features[2], x_features[2])
        style_loss = sum(mse_loss(gram_matrix(fy), sg.expand_as(gram_matrix(fy)))
                         for fy, sg in zip(y_features, style_grams))

        total_loss = content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            save_image(denormalize(y_hat.clone()), f"stylized_training_output/output_{epoch}_{i}.jpg")

        step += 1

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}")

torch.save(transformer.state_dict(), "final_styled_model.pth")
print("Model saved.")


