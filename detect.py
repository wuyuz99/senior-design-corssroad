import argparse
import os
import shutil
import time
from pathlib import Path
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from deep_sort import build_tracker
from utils.parser import get_config

def getCentroid(rect):
    x = int((rect[2] + rect[0]) / 2)
    y = int((rect[3] + rect[1]) / 2)
    return np.array([x, y])

def detect():
    out, source, weights, view_img, imgsz, show_distance, show_box = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.img_size, opt.show_distance, opt.show_box

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    frameno = 0
    mot_tracker = build_tracker(cfg, use_cuda = (device.type != 'cpu'))
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = path, '', im0s
            frameno = frameno + 1

            save_path = str(Path(out) / Path(p).name)

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                #output list
                rects = []
                confs = []
                for *xyxy, conf, cls in reversed(det):
                    confs.append(conf.cpu().numpy())
                    rect = [0, 0, 0, 0]
                    tmp = [0, 0, 0, 0]
                    for j in range(4):
                        tmp[j] = int(xyxy[j].item())
                    rect[0] = (tmp[0] + tmp[2]) / 2
                    rect[1] = (tmp[1] + tmp[3]) / 2
                    rect[2] = (tmp[2] - tmp[0])
                    rect[3] = (tmp[3] - tmp[1])

                    label = '%s %.2f' % (names[int(cls)], conf)
                    if show_box:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    rects.append(rect)
                ids = mot_tracker.update(np.array(rects), np.array(confs), im0)
                centroids = []
                for j in range(len(ids)):
                    centroid = getCentroid(ids[j][0:4])
                    objectID = ids[j][4]
                    text = "ID {}".format(objectID)
                    cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    centroids.append(centroid)
                if show_distance and len(centroids) != 0:
                    d = dist.cdist(centroids, centroids)
                    for j in range(len(centroids)):
                        for k in range(j + 1, len(centroids)):
                            cv2.line(im0, tuple(centroids[j]), tuple(centroids[k]), (0, 255, 0))
                            text = str(int(d[j, k]))
                            mid = tuple(((centroids[j] + centroids[k]) / 2).astype('int32'))
                            cv2.putText(im0, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

    print('Results saved to %s' % Path(out))
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--config_deepsort', default='configs/deep_sort.yaml', help='deep_sort.yaml path')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--show-distance', action='store_true', help='display distance')
    parser.add_argument('--show-box', action='store_true', help='display box')


    opt = parser.parse_args()
with torch.no_grad():
    detect()
