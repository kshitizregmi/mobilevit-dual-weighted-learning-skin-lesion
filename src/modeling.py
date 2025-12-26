import timm
import torch.nn as nn
from src.config import settings

def create_model(num_classes: int) -> nn.Module:
    """
    Constructs the model architecture based on configuration settings.
    
    Wraps timm.create_model to centralize architecture definitions.
    """
    model = timm.create_model(
        settings.MODEL_ARCH,
        pretrained=settings.PRETRAINED,
        num_classes=num_classes
    )
    
    return model