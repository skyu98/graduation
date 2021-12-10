import torch
from models import Darknet
from models import save_weights

cfg = './yolov3.cfg'
weights = './weights/best.pt'
img_size = 416

model = Darknet(cfg, img_size)

checkpoint = torch.load(weights, map_location='cpu')

model.load_state_dict(checkpoint['model'])

save_weights(model, './weights/jar.weights', cutoff=-1)
