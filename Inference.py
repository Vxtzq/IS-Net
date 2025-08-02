import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io
from PIL import Image
from models import ISNetDIS  # Assuming this is your custom model
from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Vxtzq/Is-Net",
    filename="isnet-general-use.pth",
    local_dir="./",  # or use any specific folder path
    local_dir_use_symlinks=False  # ensures the file is copied, not symlinked
)

print("Model downloaded to:", model_path)

# --- Config ---
input_image_path = "sprite/some_image.png"  # ← CHANGE THIS
output_path = "sprite_output/sd_output.png"
input_size = [1024, 1024]
model_filename = "isnet-general-use.pth"
repo_id = "Vxtzq/Is-Net"
# --------------

# Ensure output dir exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download model from HuggingFace if not already present
model_path = hf_hub_download(
    repo_id=repo_id,
    filename=model_filename,
    local_dir="./",
    local_dir_use_symlinks=False
)
print("Model downloaded to:", model_path)

# Load model
net = ISNetDIS()
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
else:
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()

# Load and prepare input image
im = io.imread(input_image_path)
if len(im.shape) < 3:  # Grayscale → RGB
    im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
original_shape = im.shape[:2]  # (H, W)

# Convert to tensor and resize to model input size
im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
im_tensor = F.interpolate(im_tensor.unsqueeze(0), input_size, mode="bilinear").type(torch.uint8)

# Normalize
image = im_tensor / 255.0
image = (image - 0.5) / 1.0

if torch.cuda.is_available():
    image = image.cuda()

# Inference
with torch.no_grad():
    result = net(image)[0][0]
    result = F.interpolate(result, size=original_shape, mode='bilinear', align_corners=False)
    result = result.squeeze(0).squeeze(0)

    # Normalize alpha mask to 0-255
    alpha = (result - result.min()) / (result.max() - result.min())
    alpha_np = (alpha.cpu().numpy() * 255).astype(np.uint8)

# Reconstruct RGBA image
im_pil = Image.fromarray(im).convert("RGB").resize((original_shape[1], original_shape[0]))
rgb_np = np.array(im_pil)
rgba_np = np.dstack([rgb_np, alpha_np])

# Save
Image.fromarray(rgba_np, mode="RGBA").save(output_path)
print(f"Saved output to {output_path}")
