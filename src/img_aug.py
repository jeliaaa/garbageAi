import os
import random
from PIL import Image
from torchvision import transforms

# Define image augmentation transformations
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])

def augment_images(input_dir, output_dir, num_augmentations=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            image = Image.open(img_path).convert("RGB")
            
            # Create augmented images
            for i in range(num_augmentations):
                augmented_image = augmentations(image)
                augmented_image_filename = f"{os.path.splitext(filename)[0]}_aug_{i + 1}{os.path.splitext(filename)[1]}"
                augmented_image.save(os.path.join(output_dir, augmented_image_filename))
                print(f"Saved: {augmented_image_filename} in {output_dir}")

if __name__ == "__main__":
    # Directories
    garbage_dir = "C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/data/validate/dirty"
    clean_dir = "C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/data/validate/clean"
    augmented_garbage_dir = "C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/data/validate/dirty"
    augmented_clean_dir = "C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/data/validate/clean"

    # Augment images
    augment_images(garbage_dir, augmented_garbage_dir, num_augmentations=10)
    augment_images(clean_dir, augmented_clean_dir, num_augmentations=10)

    print("Image augmentation completed!")
