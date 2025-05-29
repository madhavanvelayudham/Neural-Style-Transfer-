import argparse
import torch
from utils import load_image, save_image
from model import TransformerNet

parser = argparse.ArgumentParser()
parser.add_argument("--content", required=True, help="Path to the content image")
parser.add_argument("--output", default="output.jpg", help="Path to save the stylized image")
parser.add_argument("--model", default="final_styled_model.pth", help="Path to the trained model")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerNet().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

content = load_image(args.content).to(device)

with torch.no_grad():
    output = model(content)

save_image(output, args.output)
print(f"Saved stylized image to {args.output}")
