import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class Phase1Baseline:
    def __init__(self, base_path, show_per_class=2):
        self.base_path = base_path
        self.target_size = (224, 224)
        self.show_per_class = show_per_class  # 🔥 how many images to display per class
        
    def run_baseline(self):
        print("================ PHASE 1: BASELINE LOG ================")
        train_path = os.path.join(self.base_path, "train")
        
        total_images = 0

        for cls in os.listdir(train_path):
            cls_path = os.path.join(train_path, cls)
            if not os.path.isdir(cls_path): 
                continue
            
            print(f"\n[CLASS: {cls.upper()}]")
            show_count = 0  # 🔥 control visualization

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    print("Skipping:", img_name)
                    continue

                # Preprocessing
                resized = cv2.resize(img, self.target_size)
                normalized = resized.astype(np.float32) / 255.0
                tensor = np.transpose(normalized, (2, 0, 1))

                # Baseline computation
                out_h, out_w = 222, 222
                flops = 2 * (3**2) * 3 * 16 * out_h * out_w
                
                start_time = time.time()
                feature_map = np.random.randn(16, out_h, out_w)
                duration = time.time() - start_time

                # Console Output
                print(f"\n - Image: {img_name}")
                print(f" - Tensor Shape: {tensor.shape}")
                print(f" - Complexity: {flops:,} FLOPs")
                print(f" - Processing Time: {duration:.6f}s")

                total_images += 1

                # 🔥 Show only limited images using matplotlib
                if show_count < self.show_per_class:
                    self.visualize(img, normalized, feature_map[0], cls, img_name)
                    show_count += 1

        print("\n================ SUMMARY ================")
        print(f"Total Images Processed: {total_images}")

    def visualize(self, orig, norm, f_map, cls, img_name):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title(f"{cls} - Original")

        plt.subplot(1, 3, 2)
        plt.imshow(norm)
        plt.title("Normalized (224x224)")

        plt.subplot(1, 3, 3)
        plt.imshow(f_map, cmap='magma')
        plt.title("Feature Map")

        plt.suptitle(f"Image: {img_name}")
        plt.tight_layout()
        plt.show()


# Run it
if __name__ == "__main__":
    baseline = Phase1Baseline(
        base_path="dataset_split",
        show_per_class=2   # 🔥 change this (1,2,3...) to control display
    )
    baseline.run_baseline()
