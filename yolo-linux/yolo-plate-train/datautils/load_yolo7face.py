import cv2
import numpy as np
import  os,sys
import  numpy as np
import random
import torch

def limit_img_auto(imgin):
    img=np.array(imgin)
    sw = 1920 * 1.2
    sh = 1080 * 1.2
    h, w = tuple(list(imgin.shape)[0:2])
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    resize_ratio=sh/h
    if whratio > swhratio:
        resize_ratio=1.0*sw/w
    if resize_ratio<1:
        img=cv2.resize(imgin,None,fx=resize_ratio,fy=resize_ratio)
    return img

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def split_imkey(impath):
    imname = os.path.basename(impath)
    ext=imname.split('.')[-1]
    imkey=imname[0:len(imname)-len(ext)-1]
    return imkey,ext

def load_yololabel(txtpath):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    label_flatten_list=[]
    for line in lines:
        numbers = line.split(' ')
        num_array = np.array([float(number) for number in numbers])
        # print('num_array:',num_array)
        pts_array = np.delete(num_array[5:], np.arange(2, num_array.shape[0] - 5,3))  # remove the occlusion paramater from the GT
        num_array = np.hstack((num_array[:5], pts_array))
        label_flatten_list.append(num_array)
    return label_flatten_list

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0, kpt_label=False):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # it does the same operation as above for the key-points
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[ 2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    if kpt_label:
        num_kpts =5
        for kpt in range(num_kpts):
                if y[ 2 * kpt + 4]!=0:
                    y[2*kpt+4] = w * y[2*kpt+4] + padw
                if y[ 2 * kpt + 1 + 4] !=0:
                    y[2*kpt+1+4] = h * y[2*kpt+1+4] + padh
    return y


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def get_origin_ano():
    xywh=np.array(label_flatten[1:5])
    xywh[::2]*=w
    xywh[1::2] *= h
    xyxy=xywh2xyxy(xywh)
    xyxy=xyxy.astype(np.int32)
    print('label_flatten:',label_flatten)
    kptsx = label_flatten[5::2]*w
    kptsy = label_flatten[6::2]*h
    kptsx=kptsx.astype(np.int32)
    kptsy = kptsy.astype(np.int32)

if __name__=='__main__':
    # dataroot=r'Z:\workspace\yoloface\dataset\WIDER_train\WIDER_train\images'
    # txtroot=r'Z:\workspace\yoloface\dataset\yolov7-face-label\yolov7-face-label\train'
    dataroot=r'Z:\workspace\yoloface\dataset\label_test'
    txtroot=r'Z:\workspace\yoloface\dataset\label_test'


    ims=get_ims(dataroot)
    for impath in ims:

        imname=os.path.basename(impath)
        imkey,ext=split_imkey(impath)
        txtname=imkey+'.txt'
        txtpath=os.path.join(txtroot,txtname)

        img_const = cv2.imread(impath)
        h,w,c=img_const.shape

        print(impath,txtpath)
        label_flatten_list=load_yololabel(txtpath)

        img, ratio, pad = letterbox(img_const, 640, stride=32)
        for label_flatten in label_flatten_list:
            print('label_flatten:',label_flatten.shape)

            label_flatten[1:] = xywhn2xyxy(label_flatten[1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1],kpt_label=True)
            xyxy = np.array(label_flatten[1:5]).astype(np.int32)
            kptsx = label_flatten[5::2].astype(np.int32)
            kptsy = label_flatten[6::2].astype(np.int32)

            cv2.rectangle(img, [xyxy[0],xyxy[1]], [xyxy[2],xyxy[3]], (0, 255, 0), thickness=2)

            for k in range(0,5):
                cv2.circle(img, (kptsx[k],kptsy[k]), 2,(0, 255, 255), thickness=-1)


        cv2.imshow('img_const',limit_img_auto(img_const))

        # img, ratio, (dw, dh) = letterbox(img_const, 640, stride=32)
        cv2.imshow('img', limit_img_auto(img))

        print('img.shape:',img.shape)

        if cv2.waitKey(0)==27:
            exit(0)

    print('finish')