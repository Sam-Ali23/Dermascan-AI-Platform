import cv2
import torch


def preprocess_image(image, image_size=(256, 256)):
    image = cv2.resize(image, image_size)
    image = image / 255.0
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor