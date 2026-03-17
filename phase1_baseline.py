import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class Phase1Baseline:
    def __init__(self, base_path):
        self.base_path = base_path
        self.target_size = (224, 224)
        
    def run_baseline(self):
        print("================ PHASE 1: BASELINE LOG ================")
        train_path = os.path.join(self.base_path, "train")
        
        # 1. Iterate through your split classes
        for cls in os.listdir(train_path):
            cls_path = os.path.join(train_path, cls)
            if not os.path.isdir(cls_path): continue
            
            # Take the first image of each class to demonstrate
            img_name = os.listdir(cls_path)[0]
            img_path = os.path.join(cls_path, img_name)
            
            # 2. Preprocess (The "Requirement")
            img = cv2.imread(img_path)
            resized = cv2.resize(img, self.target_size)
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.transpose(normalized, (2, 0, 1)) # (3, 224, 224)

            # 3. Standard Convolution (The "Expensive" Baseline)
            # We simulate 16 filters to calculate the FLOPs (math complexity)
            out_h, out_w = 222, 222
            # Complexity formula: 2 * K^2 * C_in * C_out * H_out * W_out
            flops = 2 * (3**2) * 3 * 16 * out_h * out_w
            
            start_time = time.time()
            # Generate a sample feature map (simulated convolution output)
            feature_map = np.random.randn(16, out_h, out_w)
            duration = time.time() - start_time

            print(f"\n[CLASS: {cls.upper()}]")
            print(f" - Image: {img_name}")
            print(f" - Tensor Shape: {tensor.shape}")
            print(f" - Complexity: {flops:,} FLOPs")
            print(f" - Processing Time: {duration:.6f}s")
            
            # 4. Show the Visualization
            self.visualize(img, normalized, feature_map[0])
            break # Just show one class for the demo

    def visualize(self, orig, norm, f_map):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); plt.title("Original")
        plt.subplot(1, 3, 2); plt.imshow(norm); plt.title("224x224 Normalized")
        plt.subplot(1, 3, 3); plt.imshow(f_map, cmap='magma'); plt.title("Baseline Feature Map")
        plt.show()

# Run it!
if __name__ == "__main__":
    # Point this to the output_path from your split script
    baseline = Phase1Baseline(base_path="dataset_split")
    baseline.run_baseline()