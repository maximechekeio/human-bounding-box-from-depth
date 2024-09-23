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
TEST_IMG_DIR_1 = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_images_subject_1"
TEST_MASK_DIR_1 = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_masks_subject_1"
TEST_IMG_DIR_2 = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_images_subject_2"
TEST_MASK_DIR_2 = "c:\\Users\\maxime\\Documents\\Datasets\\human-bounding-box-from-depth-dataset\\test_masks_subject_2"


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
    test_loader_1, test_loader_2 = get_loaders(
        TEST_IMG_DIR_1,
        TEST_MASK_DIR_1,
        TEST_IMG_DIR_2,
        TEST_MASK_DIR_2,
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
    check_accuracy(test_loader_1, model, device=DEVICE)
    #check_accuracy(test_loader_2, model, device=DEVICE)

    # print some examples to a folder
    save_predictions_as_imgs(
        test_loader_1, model, folder="inference_images_subject_1/", device=DEVICE
    )
    #save_predictions_as_imgs(
    #    test_loader_2, model, folder="inference_images_subject_2/", device=DEVICE
    #)

if __name__ == "__main__":
    main()