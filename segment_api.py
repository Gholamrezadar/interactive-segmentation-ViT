import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision

from segment_anything import sam_model_registry, SamPredictor

from ghdtimer import Timer
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# predictor = SamPredictor(sam)

class SegmentApi():
    def __init__(self, model_type='vit_b') -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_type = model_type
        self.model = None
        self.predictor = None

        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    def load_model(self, model_path):
        self.model = sam_model_registry[self.model_type](checkpoint=model_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
    
    def process_image(self, image):
        self.predictor.set_image(image)
    
    def segment(self, location=[500, 300], mask_idx=0):
        # input_point = np.array([[500, 300]])
        input_point = np.array([location])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        )

        if len(masks) == 0:
            print("No masks found")
            return None
        elif mask_idx < len(masks):
            return masks[mask_idx]

    def info(self):
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
