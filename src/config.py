from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    # Experiment
    PROJECT_NAME: str = "skin_lesion_v2"
    SEED: int = 42
    
    # Smart Device Detection
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def DEVICE(self) -> str:
        return self._get_device()

    # Paths
    DATA_ROOT: Path = Path("data") 
    CHECKPOINT_DIR: Path = Path("checkpoints")
    
    # Model
    MODEL_ARCH: str = "mobilevitv2_100"
    IMG_SIZE: int = 256
    PRETRAINED: bool = True
    
    # Training Hyperparameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 60
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.05
    
    # Data Strategy
    SAMPLER_ALPHA: float = 0.5   
    LOSS_BETA: float = 0.999     
    EPOCH_SIZE_MULT: float = 1.5 
    
    # System
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    class Config:
        env_file = ".env"
        ignored_types = (property,)

settings = Settings()
settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)