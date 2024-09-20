import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import LidarDataset
import numpy as np
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640

train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )

val_transforms = A.Compose(
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

image_dir = "data/train_images/"
mask_dir = "data/train_masks/"

train_dataset = LidarDataset(image_dir, mask_dir, train_transforms)

# Test if data can be loaded correctly
sample_image, sample_mask = train_dataset[0]  # Load the first sample

print(f"Image shape: {sample_image.shape}")
print(f"Mask shape: {sample_mask.shape}")
print(f"Unique mask values: {np.unique(sample_mask)}")



# Visualize the image and mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze().permute(1,2,0), cmap="viridis")
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("Mask")

plt.show()