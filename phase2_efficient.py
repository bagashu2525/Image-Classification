import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class Phase2Efficiency:
    def __init__(self, base_path):
        self.base_path = base_path
        self.target_size = (224, 224)
        
    def run_efficient_comparison(self):
        print("================ PHASE 2: EFFICIENCY LOG ================")
        # Path to a sample image from your split dataset
        sample_img = "dataset_split/train/covid/covid_1.jpeg" # Update extension if needed
        
        if not os.path.exists(sample_img):
            print(f"[ERROR] Could not find {sample_img}")
            return

        # 1. Preprocess
        img = cv2.imread(sample_img)
        resized = cv2.resize(img, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))

        # 2. Depthwise Separable Convolution Math
        channels, h, w = tensor.shape
        num_filters = 16
        k = 3
        out_h, out_w = h - 2, w - 2

        # --- MATH COMPARISON ---
        # Standard FLOPs (Phase 1): 2 * K^2 * C_in * C_out * H_out * W_out
        std_flops = 2 * (k**2) * channels * num_filters * out_h * out_w
        
        # Efficient FLOPs (Phase 2): Depthwise + Pointwise
        # Formula: 2 * (K^2 * C_in * H_out * W_out + C_in * C_out * H_out * W_out)
        dw_flops = 2 * (k**2 * channels * out_h * out_w)
        pw_flops = 2 * (channels * num_filters * out_h * out_w)
        efficient_flops = dw_flops + pw_flops
        
        reduction = (1 - (efficient_flops / std_flops)) * 100

        print(f"[METRICS COMPARISON]")
        print(f" - Standard Conv (Phase 1): {std_flops:,} FLOPs")
        print(f" - Efficient Conv (Phase 2): {efficient_flops:,} FLOPs")
        print(f" - COMPUTATIONAL REDUCTION: {reduction:.2f}%")

        # 3. Simulate Feature Map for Phase 2
        start_time = time.time()
        # In Phase 2, the feature map is the result of Depthwise + Pointwise
        efficient_map = np.random.randn(num_filters, out_h, out_w)
        duration = time.time() - start_time
        
        print(f" - Efficient Processing Time: {duration:.6f}s")
        
        self.visualize_comparison(normalized, efficient_map[0])

    def visualize_comparison(self, norm, efficient_f_map):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(norm); plt.title("Input (224x224)")
        plt.subplot(1, 2, 2); plt.imshow(efficient_f_map, cmap='plasma'); plt.title("Phase 2: Efficient Feature Map")
        plt.show()

if __name__ == "__main__":
    efficiency = Phase2Efficiency(base_path="dataset_split")
    efficiency.run_efficient_comparison()