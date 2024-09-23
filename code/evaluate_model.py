import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# Hyperparameters etc. (need to load val data because get loaders is used in the main function)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = True
VAL_IMG_DIR = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\val_images" 
VAL_MASK_DIR = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\val_masks"
TEST_IMG_DIR = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_images"
TEST_MASK_DIR = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_masks"

def main():
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # Get the loaders of val and test data, but only use the test loader
    val_loader, test_loader = get_loaders(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        test_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        checkpoint = torch.load("my_checkpoint.pth.tar")
        load_checkpoint(checkpoint, model)
        
    # check accuracy
    check_accuracy(test_loader, model, device=DEVICE)

    # print some examples to a folder
    save_predictions_as_imgs(
        test_loader, model, folder="inference_images/", device=DEVICE
    )

if __name__ == "__main__":
    main()