import torch
from app.multitask_model import MultiTaskResNetUNet



class ModelLoader:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiTaskResNetUNet().to(self.device)

        self.model.load_state_dict(
            torch.load("skin_multitask_ai.pth", map_location="cpu")
        )

        self.model.eval()

    def get_model(self):
        return self.model