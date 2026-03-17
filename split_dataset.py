import os
import shutil

# Source dataset path
dataset_path = "Dataset"

# Output folder
output_path = "dataset_split"

train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

# Classes
classes = ["covid", "normal", "pneumonia"]

for cls in classes:

    src_folder = os.path.join(dataset_path, cls)

    train_folder = os.path.join(train_path, cls)
    test_folder = os.path.join(test_path, cls)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = os.listdir(src_folder)

    # First 5 images for train
    train_images = images[:5]

    # Next 5 images for test
    test_images = images[5:10]

    # Copy train images
    for img in train_images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(train_folder, img)
        shutil.copy(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(test_folder, img)
        shutil.copy(src, dst)

print("Dataset successfully split!")
