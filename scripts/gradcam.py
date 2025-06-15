import os
import torch
import numpy as np
import cv2

from new_model_train import EfficientNet  # our EfficientNet model
from data_preprocessing import dataPreprocessing  # Data Preprocessing

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
    return overlay, image

#To create comparison between original image and grad-cam image
def create_comparison_image(original, gradcam_overlay):
    comparison = np.hstack((original, gradcam_overlay))
    return comparison


if __name__ == "__main__":
    image_path = "PrivateTest_10629254.jpg"  # Image to be evaluated
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load model and weights (parameters)
    model = EfficientNet(num_classes=7)
    model.load_state_dict(torch.load("model_2.pth", map_location="cpu"))

    # Image Preprocessing
    input_tensor = dataPreprocessing(image_path).unsqueeze(0)

    # Select target layer (Choose Last Convolutional Block)
    target_layer = model.blocks[-1]
    grad_cam = GradCAM(model, target_layer)

    # Generate gradcam heatmap
    heatmap = grad_cam.compute_gradcam(input_tensor)
    gradcam_image, original_image = overlay_heatmap(heatmap, image_path)

    # Create comparison image
    comparison_image = create_comparison_image(original_image, gradcam_image)

    # Save images to "outputs" folder
    gradcam_path = os.path.join(output_dir, "gradcam.jpg")
    comparison_path = os.path.join(output_dir, "gradcam_comparison.jpg")
    cv2.imwrite(gradcam_path, gradcam_image)
    cv2.imwrite(comparison_path, comparison_image)

    print(f"GradCam output saved at {gradcam_path}")
    print(f"Comparison saved at {comparison_path}")

    # Display GradCam
    cv2.imshow("GradCAM Comparison", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_cam.remove_hooks()
