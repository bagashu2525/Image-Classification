import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class Phase2Efficiency:
    def __init__(self, base_path, show_per_class=1):
        self.base_path = base_path
        self.target_size = (224, 224)
        self.show_per_class = show_per_class  # 🔥 control visualization
        
    def run_efficient_comparison(self):
        print("================ PHASE 2: EFFICIENCY LOG ================")

        train_path = os.path.join(self.base_path, "train")
        total_images = 0

        for cls in os.listdir(train_path):
            cls_path = os.path.join(train_path, cls)
            if not os.path.isdir(cls_path):
                continue

            print(f"\n[CLASS: {cls.upper()}]")
            show_count = 0

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)

                if not os.path.exists(img_path):
                    print("Skipping:", img_name)
                    continue

                # 1. Preprocess
                img = cv2.imread(img_path)
                if img is None:
                    print("Skipping (invalid image):", img_name)
                    continue

                resized = cv2.resize(img, self.target_size)
                normalized = resized.astype(np.float32) / 255.0
                tensor = np.transpose(normalized, (2, 0, 1))

                # 2. Depthwise Separable Convolution Math
                channels, h, w = tensor.shape
                num_filters = 16
                k = 3
                out_h, out_w = h - 2, w - 2

                # Standard FLOPs
                std_flops = 2 * (k**2) * channels * num_filters * out_h * out_w

                # Efficient FLOPs
                dw_flops = 2 * (k**2 * channels * out_h * out_w)
                pw_flops = 2 * (channels * num_filters * out_h * out_w)
                efficient_flops = dw_flops + pw_flops

                reduction = (1 - (efficient_flops / std_flops)) * 100

                print(f"\n - Image: {img_name}")
                print(f"   Standard Conv: {std_flops:,} FLOPs")
                print(f"   Efficient Conv: {efficient_flops:,} FLOPs")
                print(f"   Reduction: {reduction:.2f}%")

                # 3. Simulate Feature Map
                start_time = time.time()
                efficient_map = np.random.randn(num_filters, out_h, out_w)
                duration = time.time() - start_time

                print(f"   Efficient Processing Time: {duration:.6f}s")

                total_images += 1

                # 🔥 Controlled visualization
                if show_count < self.show_per_class:
                    self.visualize_comparison(normalized, efficient_map[0], cls, img_name)
                    show_count += 1

        print("\n================ SUMMARY ================")
        print(f"Total Images Processed: {total_images}")

    def visualize_comparison(self, norm, efficient_f_map, cls, img_name):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(norm)
        plt.title(f"{cls} - Input (224x224)")

        plt.subplot(1, 2, 2)
        plt.imshow(efficient_f_map, cmap='plasma')
        plt.title("Efficient Feature Map")

        plt.suptitle(f"Image: {img_name}")
        plt.tight_layout()
        plt.show()


# Run it
if __name__ == "__main__":
    efficiency = Phase2Efficiency(
        base_path="dataset_split",
        show_per_class=1   # 🔥 change to control plots
    )
    efficiency.run_efficient_comparison()
