from __future__ import division
import torch
import torch.nn as nn
import os
import cv2
import time
import numpy as np

from utils.utils import BBox, drawLandmark_multiple
from utils.general import letterbox, check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from models.experimental import attempt_load
from models.basenet import MobileNet_GDConv


def load_model(cfg):
    # MobileNet:
    model = MobileNet_GDConv(136)
    model = torch.nn.DataParallel(model)

    # load pretrained model:
    checkpoint = torch.load('./checkpoint/weights/mobilenet.pth.tar', map_location=cfg.device)
    model.load_state_dict(checkpoint['state_dict'])

    # set mode to be eval
    model = model.eval()

    return model


def load_model_det(cfg):
    model = attempt_load('./checkpoint/det_weights/yolov5s-face.pt', map_location=cfg.device)
    return model


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride
    return max_size


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def detect(cfg, model, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = 640

    # original size
    if imgsz <= 0:
        imgsz = dynamic_resize(img0.shape)

    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(cfg.device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # TensorShape must to be (b, c, h, w):
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # YOLO forward:
    pred = model(img, augment=cfg.augment)[0]

    # Apply NMS:
    pred = non_max_suppression_face(pred, cfg.conf_thres, cfg.iou_thres)[0]  # [n,16],n is number people

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(cfg.device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(cfg.device)  # normalization gain landmarks
    boxes = []
    h, w, c = img0.shape

    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()

        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()  # only one category: face
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2 - x1, y2 - y1, conf])

    return boxes


def mouth_aspect_ratio(mouth):
    # 垂直点位:
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


class Config(object):
    def __init__(self):
        self.conf_thres = 0.02
        self.iou_thres = 0.5
        self.augment = False
        self.mean = np.asarray([0.485, 0.456, 0.406])
        self.std = np.asarray([0.229, 0.224, 0.225])
        self.DET_THRESH = 0.65  # detection scroe threshold
        self.SCALE = 1.2  # boxes scale factor
        self.MAR_THRESH = 0.9  # open mouthe threshold
        self.out_size = 224
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Pipeline(nn.Module):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.cfg = Config()
        self.model = load_model(self.cfg)
        self.model_det = load_model_det(self.cfg)

    def predict(self, img):
        height, width, _ = img.shape
        with torch.no_grad():
            start1 = time.time()

            # YOLO fordward:
            faces = detect(self.cfg, self.model_det, img)  # [[x1, y1, x2-x1, y2-y1, array(score, dtype=float32)],[...]]

            end1 = time.time()

            faces_data = []

            if len(faces) == 0:
                return faces_data

            for k, face in enumerate(faces):
                if face[4] < self.cfg.DET_THRESH:
                    continue

                x1 = face[0]
                y1 = face[1]
                w = face[2]
                h = face[3]
                det_score = face[4]
                size = int(min([w, h]) * self.cfg.SCALE)
                cx = x1 + w // 2
                cy = y1 + h // 2
                x1 = cx - size // 2
                x2 = x1 + size
                y1 = cy - size // 2
                y2 = y1 + size

                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)
                new_bbox = list(map(int, [x1, x2, y1, y2]))
                new_bbox = BBox(new_bbox)
                cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]  # ROIs

                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)

                cropped_face = cv2.resize(cropped, (self.cfg.out_size, self.cfg.out_size))

                if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                    continue

                test_face = cropped_face.copy()
                test_face = test_face / 255.0  # Normalize
                test_face = (test_face - self.cfg.mean) / self.cfg.std  # Standardize
                test_face = test_face.transpose((2, 0, 1))  # Channel dimension
                test_face = test_face.reshape((1,) + test_face.shape)  # unsqueeze(0): CHW2BCHW

                input = torch.from_numpy(test_face).float()  # ToTensor && set dtype to be float32
                input = torch.autograd.Variable(input)

                # MobileNet forward:
                start2 = time.time()
                landmark = self.model(input).cpu().data.numpy()
                end2 = time.time()

                landmark = landmark.reshape(-1, 2)
                landmark = new_bbox.reprojectLandmark(landmark)  # len=68
                mouth_points = landmark[48:68]  # len=20
                mouth_ear = mouth_aspect_ratio(mouth_points)
                mouth_points = [mouth_points[0], mouth_points[2], mouth_points[4], mouth_points[6], mouth_points[7],
                                mouth_points[9]]
                flag = 1 if mouth_ear > self.cfg.MAR_THRESH else 0
                data = {
                    'face_time': round(end1 - start1, 6),
                    'face_score': round(det_score.tolist(), 3),
                    'face_bbox': new_bbox,
                    'mouth_time': round(end2 - start2, 6),
                    'mouth_points': mouth_points,
                    'mouth_score': round(mouth_ear, 3),
                    'mouth_success': flag,
                }
                faces_data.append(data)
            return faces_data
