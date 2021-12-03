from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@torch.no_grad()
def findBox(img0):
    img_size = 416 # (320, 192) or (416, 256) or (608, 352) for (height, width)
    cfg = '../yolo/yolov3.cfg'
    weights = '../yolo/weights/best.pt'
    conf_thres = 0.2
    nms_thres = 0.5

    view_img = False

    # Initialize
    device = torch_utils.select_device(device='cpu')
    
    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()
    
    # Padded resize
    img = letterbox(img0, new_shape = img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = ['jar']
    colors = [255, 120, 120]

    # Run inference
    # t0 = time.time()

    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, nms_thres)

    # Process detections
    for _, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # print("Jar Found.")
            
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, _, cls in det:
                res = [int(i.item()) for i in xyxy]

                if view_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])

        if det is None:
            # print("Jar Not Found.")
            return (0, 0, 0, 0)

        # Stream results
        if view_img:
            cv2.imshow('res', img0)

    # print('Done. (%.3fs)' % (time.time() - t0))
    return tuple(res[:4])



# img0 = cv2.imread('../imgs/input_imgs/origin.jpg')
# findBox(img0)