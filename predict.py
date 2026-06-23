import torch
from PIL import Image
from torchvision import transforms

from model import get_model
from preprocess import preprocess_image


device = torch.device("cpu")


model = get_model()

model.load_state_dict(
    torch.load(
        "devanagari_resnet18.pth",
        map_location=device
    )
)

model.eval()