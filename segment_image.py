import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_api import SegmentApi
from ghdtimer import Timer

# Load image
image = cv2.imread('images/truck_low.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Inference
segment_api = SegmentApi(model_type='vit_b')

timer = Timer()

print("\nLoading model...")
timer.tick()
segment_api.load_model(model_path='models/sam_vit_b_01ec64.pth')
timer.tock()

print("\nProcessing image...")
timer.tick()
segment_api.process_image(image)
timer.tock()

mask = np.zeros_like(image)
org_image = image.copy()

# Mouse callback function to get location
def mouseCallback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image = org_image.copy()
        cv2.circle(image,(x,y),5,(0,255,0),-1)
        print("\nSegmenting...")
        timer.tick()
        mask = segment_api.segment(location=[x, y])
        timer.tock()
        mask = segment_api.segment(location=[x, y])
        mask = mask.astype(np.uint8)
        mask = mask * 255

        cv2.imshow("Mask", mask)
        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouseCallback)

# Main Loop
cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
while(1):
    key=cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()