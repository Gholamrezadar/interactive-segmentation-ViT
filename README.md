# Interactive Segmentaion ViT

This is a demo of []() from meta.

## Usage

```shell
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

Other dependencies:

```shell
opencv-python
torch
torchvision
matplotlib
numpy
ghdtimer
```

Download the pretrained model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it in the `models` folder.
If you want to use bigger models, you can download them from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) and rename the model path in `segment_image.py`. Dont forget to set the model_type too(vit_h, vit_b, ...).
## Demo

```shell
python segment
