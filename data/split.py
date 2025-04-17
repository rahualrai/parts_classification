import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, output_dir, test_size=0.2):
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            for img in train_images:
                shutil.copy(img, train_class_dir)
            for img in test_images:
                shutil.copy(img, test_class_dir)

            print(f"Class '{class_name}' split into {len(train_images)} train and {len(test_images)} test images.")

if __name__ == "__main__":
    base_dir = "data/parts"
    output_dir = "data/split_parts"
    split_dataset(base_dir, output_dir, test_size=0.2)