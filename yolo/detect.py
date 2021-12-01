import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@torch.no_grad()
def detect(imgPath):
    img_size = 416 # (320, 192) or (416, 256) or (608, 352) for (height, width)
    cfg = './yolov3.cfg'
    weights = 'weights/best.pt'
    data = 'data/jar.data'
    conf_thres = 0.2
    nms_thres = 0.5
    out = './output'
    save_txt=True, save_img=True, view_img = False

    # Initialize
    device = torch_utils.select_device(device='cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
        
    # read img
    img0 = cv2.imread(imgPath)  # BGR
    assert img0 is not None, 'Image Not Found ' + imgPath
           
    # Padded resize
    img = letterbox(img0, new_shape = img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = ['jar']
    colors = [[random.randint(0, 255) for _ in range(3)]]

    # Run inference
    t0 = time.time()
    t = time.time()

    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, nms_thres)

    # Process detections
    for _, det in enumerate(pred):  # detections per image
        p, s, im0 = imgPath, '', img0

        save_path = str(Path(out) / Path(p).name)

        print(save_path)

        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                print(classes[int(c)])

            # Write results
            for *xyxy, conf, _, cls in det:
                if save_txt:  # Write to file
                    #输出的txt的每一行前4个为：左上x，左上y，右下x，右下y
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        if det is None:
            with open(save_path + '.txt', 'a') as file:
                file.write("None")
            print("None")
            
        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Stream results
        if view_img:
            cv2.imshow(p, im0)

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


with torch.no_grad():
    detect()
