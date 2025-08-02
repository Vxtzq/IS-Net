import os
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from models import ISNetDIS  # assuming your model class here
import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *
if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="Vxtzq/Is-Net",
        filename="isnet-general-use.pth",
        local_dir="./",  # or use any specific folder path
        local_dir_use_symlinks=False  # ensures the file is copied, not symlinked
    )
    
    print("Model downloaded to:", model_path)
    dataset_path="sprite"  # Your dataset path
    model_path="isnet-general-use.pth"  # the model path
    result_path="sprite_output"  # The folder path that you want to save the results
    input_size=[1024,1024]
    net=ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()

    im_list = (glob(dataset_path+"/*.jpg") + glob(dataset_path+"/*.JPG") +
               glob(dataset_path+"/*.jpeg") + glob(dataset_path+"/*.JPEG") +
               glob(dataset_path+"/*.png") + glob(dataset_path+"/*.PNG") +
               glob(dataset_path+"/*.bmp") + glob(dataset_path+"/*.BMP") +
               glob(dataset_path+"/*.tiff") + glob(dataset_path+"/*.TIFF"))

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("Processing:", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:  # grayscale to RGB
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            im_shp = im.shape[0:2]
            im_shp = tuple(int(x) for x in im_shp)
            print("im_shp:", im_shp, type(im_shp))
            # Convert to tensor and resize to input_size
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)

            image = torch.divide(im_tensor, 255.0)
            # Normalize
            image = (image - 0.5) / 1.0  # same as normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
            
            if torch.cuda.is_available():
                image = image.cuda()

            # Run model to get mask (alpha)
            result = net(image)
            
            result = result[0][0]
            print("result shape before interpolate:", result.shape)
            # Interpolate to original image size
            result = F.interpolate(result, size=im_shp, mode='bilinear', align_corners=False)
            # Remove batch and channel dims: [1, 1, H, W] -> [H, W]
            result = result.squeeze(0).squeeze(0)

            # Normalize mask to 0-1
            ma = torch.max(result)
            mi = torch.min(result)
            alpha = (result - mi) / (ma - mi)

            # Prepare final RGBA image
            alpha_np = (alpha.cpu().numpy() * 255).astype(np.uint8)  # HxW alpha mask

            # Resize original image to original shape (in case)
            im_pil = Image.fromarray(im).convert("RGB").resize((im_shp[1], im_shp[0]))  # PIL uses (width, height)
            rgb_np = np.array(im_pil)

            # Combine RGB + alpha
            rgba_np = np.dstack([rgb_np, alpha_np])

            im_name = os.path.splitext(os.path.basename(im_path))[0]
            out_path = os.path.join(result_path, "sd_output.png")
            Image.fromarray(rgba_np, mode="RGBA").save(out_path)


