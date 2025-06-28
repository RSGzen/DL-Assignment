import os
import torch
import numpy as np
import cv2
import random
from glob import glob

from new_model_train import EfficientNet  # our EfficientNet model
from data_processing import dataPreprocessing # Data Preprocessing

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.all_hooks = []
        self.hooks()

    # Hooks for forward and backward pass
    def hooks(self):
        def forward_hook(_, __, output):  # store output activation map results only, ignores input & module
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):  # store gradient of loss output; ignores input & module
            self.gradients = grad_output[0].detach()

        # To register forward_hook to capture output activations map
        self.all_hooks.append(self.target_layer.register_forward_hook(forward_hook))

        # To register backward_hook to capture output gradients
        self.all_hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    # To generate GradCam Heatmap
    def compute_gradcam(self, input_tensor, class_idx=None):
        # Forward pass the input image to get prediction
        output = self.model(input_tensor)

        # Choose predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()  # Get integer index value of the class with highest predicted score

        self.model.zero_grad()  # Clear all old gradients
        class_score = output[0, class_idx]  # Get class score
        class_score.backward()  # Backward pass (compute gradients)

        # Compute average gradient weight for each channel
        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])

        activations = self.activations[0]

        # Multiply each activation channel by its average gradient weight
        # To highlight most important feature for the class
        for i in range(len(pooled_grad)):
            activations[i, :, :] *= pooled_grad[i]

        heatmap = activations.mean(dim=0).cpu().numpy()  # To calculate the mean of weighted activation
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU
        heatmap /= (heatmap.max() + 1e-8)  # Normalize heatmap to [0,1] for visualization

        return heatmap

    # clear all hooks to prevent memory leaks
    def remove_hooks(self):
        for hooks in self.all_hooks:
            hooks.remove()


# Overlay heatmap onto test image
def overlay_heatmap(heatmap, image_path, alpha=0.5):
    image = cv2.imread(image_path)

    # Resize images to have the same pixels (224*224)
    image = cv2.resize(image, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)  # Converts heatmap values from range [0,1] to [0, 255] & to 8-bit integers

    # apply heatmap onto image
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)
    return overlay

# Generate Gradcam for each class
def generate_grid(gradcam_images, labels, rows, cols):
    height, width, _ = gradcam_images[0].shape
    label_width = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)

    grid_image = np.zeros((rows * height, label_width + cols * width, 3), dtype=np.uint8)

    for row in range(rows):
        y_pos = row * height + height // 2 + 10
        label = labels[row]
        cv2.putText(grid_image, label, (10, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    for idx, img in enumerate(gradcam_images):
        row = idx // cols
        col = idx % cols
        y_start = row * height
        x_start = label_width + col * width
        grid_image[y_start:y_start + height, x_start:x_start + width] = img

    return grid_image


if __name__ == "__main__":
    test_folders = ["neutral", "happy", "sad", "angry", "disgust", "fear", "surprise"]
    base = "raw_data/test"
    test_images = []  # Images to be evaluated
    emotion_labels = []

    # Get 6 images per class
    for i in test_folders:
        test_path = os.path.join(base, i)
        image_list = glob(os.path.join(test_path, "*.jpg"))
        selected_images = random.sample(image_list, min(6, len(image_list)))
        test_images.extend(selected_images)
        emotion_labels.extend([i.capitalize()] * len(selected_images))  # e.g. ["Happy", "Happy", "Happy"]

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = EfficientNet(num_classes=7)
    model.load_state_dict(torch.load("efficientnetb0.pth", map_location="cpu"))
    model.eval()

    # Select target layer (Choose Last Convolutional Block)
    target_layer = model.blocks[-1]
    grad_cam = GradCAM(model, target_layer)

    # Generate gradcam heatmap
    gradcam_images = []

    for path in test_images:
        # Image Preprocessing
        input_tensor = dataPreprocessing(path).unsqueeze(0)
        # Generate heatmap
        heatmap = grad_cam.compute_gradcam(input_tensor)
        overlay = overlay_heatmap(heatmap, path)
        if overlay is not None:
            gradcam_images.append(overlay)

    row_labels = test_folders
    row_labels = [label.capitalize() for label in row_labels]  # Capitalized names

    # Save images to "outputs" folder
    grid_image = generate_grid(gradcam_images, labels = row_labels, rows=7, cols=6)
    grid_output_path = os.path.join(output_dir, "gradcam.jpg")
    cv2.imwrite(grid_output_path, grid_image)
    print(f"GradCam saved at {grid_output_path}")

    # Display GradCam
    cv2.imshow("GradCAM", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_cam.remove_hooks()