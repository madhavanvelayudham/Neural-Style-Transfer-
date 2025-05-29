import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import TransformerNet

def resize_image(image: Image.Image, size=(256, 256)) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)

@st.cache_resource
def load_model():
    model = TransformerNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("final_styled_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess(image, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(tensor.device)
    return tensor * std + mean

def tensor_to_pil(tensor):
    tensor = denormalize(tensor).clamp(0, 1)
    tensor = tensor.squeeze(0).cpu()  
    return transforms.ToPILImage()(tensor)


st.set_page_config(page_title="Style Transfer", layout="centered")
st.title("🎨 Neural Style Transfer")
st.caption("Upload an image → Resize → Stylize using your trained model.")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    resized_image = resize_image(input_image, size=(256, 256))

    st.image(resized_image, caption="Resized Input", use_column_width=True)

    if st.button("✨ Stylize"):
        with st.spinner("Applying style..."):
            model, device = load_model()
            input_tensor = preprocess(resized_image, device)

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_image = tensor_to_pil(output_tensor)

            st.image(output_image, caption="Stylized Output", use_column_width=True)

            output_image.save("stylized_output.jpg")
            with open("stylized_output.jpg", "rb") as f:
                st.download_button("📥 Download Output", f, "stylized_output.jpg", mime="image/jpeg")
