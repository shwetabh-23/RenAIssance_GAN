import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F

def pil_to_cv2(image):
    """Convert a PyTorch tensor (normalized between 0-1) to OpenCV format (uint8 BGR)."""
    if isinstance(image, torch.Tensor):
        image = image.squeeze().detach().cpu().numpy()  # Convert to NumPy
        
        # Ensure it's in (H, W, C) format
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 2:  # Grayscale case (H, W)
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
        
        # Scale to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # Convert to OpenCV format (BGR)
    if image.shape[-1] == 3:  # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image
def get_text_mask(image):
    # Read the image
    image = pil_to_cv2(image)  # Convert PIL image to OpenCV format
    # image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    # Dilate the edges to make the text regions more visible
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    return dilated, image

def get_bounding_boxes(mask):
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))
    bounding_boxes = merge_bounding_boxes(bounding_boxes)
    return bounding_boxes

def draw_bounding_boxes(image, bounding_boxes):
    image_with_boxes = image.copy()
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_with_boxes

def merge_bounding_boxes(bounding_boxes):
    if not bounding_boxes:
        return None  # No bounding boxes

    x_min = min(box[0] for box in bounding_boxes)
    y_min = min(box[1] for box in bounding_boxes)
    x_max = max(box[2] for box in bounding_boxes)
    y_max = max(box[3] for box in bounding_boxes)

    return (x_min, y_min, x_max, y_max)

def generate_mask(image_shape, bounding_boxes):
    """
    Creates a binary mask of the same size as the image, with 1s in the region of interest.
    
    Args:
    - image_shape: (H, W) shape of the image.
    - bounding_boxes: List of (x_min, y_min, x_max, y_max) tuples.
    
    Returns:
    - mask: Binary mask of shape (H, W), where ROI is 1 and the rest is 0.
    """
    mask = np.zeros(image_shape[:2], dtype=np.float32)  # Initialize mask with zeros
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        mask[y_min:y_max, x_min:x_max] = 1  # Set ROI to 1
    return mask

def compute_ssim_loss(image1_path, image2_path):
    # Get bounding boxes from the first image
    text_mask1, image1 = get_text_mask(image1_path)
    _, image2 = get_text_mask(image2_path)
    bounding_boxes = get_bounding_boxes(text_mask1)
    breakpoint()
    ssim_scores = []
    roi_images1 = []
    roi_images2 = []
    for (x_min, y_min, x_max, y_max) in [bounding_boxes]:
        roi1 = image1[y_min:y_max, x_min:x_max]
        roi2 = image2[y_min:y_max, x_min:x_max]

        roi_images1.append(roi1)

        roi_images2.append(roi2)

        # Convert to grayscale if needed
        roi1_gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        breakpoint()
        win_size = min(roi1_gray.shape[0], roi1_gray.shape[1], 7)  # Ensures it's ≤ 7 but odd
        win_size = win_size if win_size % 2 == 1 else win_size - 1  # Ensure it's odd
        if win_size < 2:
            # print("Warning: ROI is too small for SSIM calculation, skipping...")
            ssim_scores.append(0)
            continue
        score = ssim(roi1_gray, roi2_gray, win_size=win_size)
        # Compute SSIM between corresponding regions
        # score = ssim(roi1_gray, roi2_gray)
        ssim_scores.append(score)
    
    # Compute average SSIM loss
    avg_ssim_loss = 1 - np.mean(ssim_scores)
    
    return avg_ssim_loss

def compute_l1_loss(image1, image2):
    """
    Computes the L1 loss between two images (ROIs).
    
    Args:
        image1 (np.ndarray): First image region.
        image2 (np.ndarray): Second image region.

    Returns:
        torch.Tensor: Scalar L1 loss.
    """
    assert image1.shape == image2.shape, "ROIs must have the same shape"

    # Convert images to tensors
    # image1_tensor = torch.tensor(image1, dtype=torch.float32) 
    # image2_tensor = torch.tensor(image2, dtype=torch.float32)
    # breakpoint()
    # Compute L1 loss
    return F.l1_loss(image1, image2)

def save_tensor_as_image(tensor, filename="mask_image.png"):
    """
    Saves a PyTorch image tensor (normalized between -1 and 1) as a PNG file.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W).
        filename (str): Name of the output file.
    """
    # Ensure tensor is detached and moved to CPU
    tensor = tensor.detach().cpu()

    # Normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Convert to NumPy and move channels to last dimension
    image = tensor.permute(1, 2, 0).numpy()  # Shape: (H, W, C)

    # Clip values to ensure they remain in valid range
    image = np.clip(image, 0, 1)

    # Save using Matplotlib
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    
def compute_l1_loss_patches(input_image1, input_image2):
    # Extract ROIs and compute L1 loss
    all_avg_l1_losses = []
    images = input_image1.shape[0]
    for i in range(images):
        image_1 = input_image1[i].unsqueeze(0)
        image_2 = input_image2[i].unsqueeze(0)
        text_mask1, image1 = get_text_mask(image_1)
        _, image2 = get_text_mask(image_2)
        bounding_boxes = get_bounding_boxes(text_mask1)
        mask = generate_mask(image_1.shape[2:], [bounding_boxes])  # Generate mask
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=image_2.device).unsqueeze(0)  # Shape: (1, H, W)
        masked_image1 = image_1 * mask_tensor  # (3, H, W)
        masked_image2 = image_2 * mask_tensor  # (3, H, W)
        
        # breakpoint()
        loss = torch.nn.functional.l1_loss(masked_image1, masked_image2)
        all_avg_l1_losses.append(loss)
        # for (x_min, y_min, x_max, y_max) in [bounding_boxes]:  # Iterate over all bounding boxes
        #     roi1 = image1[y_min:y_max, x_min:x_max]
        #     roi2 = image2[y_min:y_max, x_min:x_max]
        #     roi1 = torch.from_numpy(roi1).float().to('cuda')
        #     roi2 = torch.from_numpy(roi2).float().to('cuda')
        #     # roi1.requires_grad = input_image2.requires_grad 
        #     breakpoint()
        #     roi2.requires_grad = input_image2.requires_grad
        #     # breakpoint()
        #     # Ensure both ROIs have the same shape
        #     if roi1.shape == roi2.shape and roi1.shape[0] > 0:
        #         loss = compute_l1_loss(roi1, roi2)
        #         l1_losses.append(loss)

        # Compute average L1 loss over all bounding boxes
        # if l1_losses:
        #     avg_l1_loss = sum(l1_losses) / len(l1_losses)
        #     all_avg_l1_losses.append(avg_l1_loss)
    return torch.stack(all_avg_l1_losses).mean()

def show_image(image, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    # plt.savefig()
    plt.show()

def batch_ssim_loss(image_1_batch, image_2_batch) : 
    
    batch_len = image_1_batch.shape[0]
    total_loss = 0.0
    for i in range(batch_len) : 
        image_1 = image_1_batch[i].unsqueeze(0)
        image_2 = image_2_batch[i].unsqueeze(0)
        loss = compute_ssim_loss(image_1, image_2)
        total_loss += loss
    avg_loss = total_loss / batch_len
    return avg_loss

if __name__ == "__main__":
    
    # Example usage
    image1_path = "/data/train_GAN/processed_data/images/2.png"  # Replace with your first image path
    image2_path = "/data/train_GAN/processed_data/images/4.png"  # Replace with your second image path
    loss = compute_ssim_loss(image1_path, image2_path)
    print("SSIM Loss:", loss)