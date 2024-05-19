import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import yaml

# from utils.datasets import *
from utils.datasets_debug import *


def xywh2xyxy_custom(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
    return y
def xywh2xyxy_custom_p(x):
    # y =  np.copy(x)
    y=np.array(x)
    y[2:4]=y[0:2]

    y[0]-=x[2]*0.5
    y[1]-=x[3]*0.5

    y[2]+=x[2]*0.5
    y[3]+=x[3]*0.5
    return y


def build_targets(targets):
    na=3
    kpt_label=None
    nl=3
    nkpt=5
    self_anchors=np.array( 
    [   [4,5,  6,8,  10,12],  # P3/8
        [15,19,  23,30,  39,52],  # P4/16
        [72,97,  123,164,  209,297],  # P5/32
    ]
    )

    self_anchors = torch.tensor(self_anchors).float().view(nl, -1, 2)
    self_anchors[0]=self_anchors[0] /8
    self_anchors[1]=self_anchors[1] /16
    self_anchors[2]=self_anchors[2] /32

    # print('self_anchors.shape:',self_anchors.shape)

    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, tkpt, indices, anch = [], [], [], [], []
    if kpt_label:
        gain = torch.ones(kpt_label*2+7, device=targets.device).long()  # normalized to gridspace gain
    else:
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    # 先将targets重复na份，让anchor的targets组合起来方便匹配。

    # print('targets.shape:',targets.shape)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    pshape=[(1,15,80,80),(1,15,40,40),(1,15,20,20)]
    for i in range(nl):
        anchors = self_anchors[i]
        if kpt_label:
            gain[2:kpt_label*2+6] = torch.tensor(pshape[i])[(kpt_label+2)*[3, 2]]  # xyxy gain
        else:
            gain[2:6] = torch.tensor(pshape[i])[[3, 2, 3, 2]]  # xyxy gain

        # print('gain[2:6] :',gain[2:6].shape,gain[2:6])
        # exit(0)

        # Match targets to anchors
        t = targets * gain
        if nt:
            # r = t[:, :, 4:6] / anchors[:, None]
            # print(t[:, :, 4:6].shape,anchors[:, None].shape,r.shape)
            # exit(0)
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            # print(r)
            j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse


            # exit(0)

            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        print('gi:',gi)
        print('gi.clamp_(0, gain[2] - 1):',gi.clamp_(0, gain[2] - 1))

        # Append
        a = t[:, -1].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        if kpt_label:
            for kpt in range(nkpt):
                t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6+2*kpt: 6+2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
            tkpt.append(t[:, 6:-1])
        # print('a:',a)
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, tkpt, indices, anch



def _make_grid(nx=20, ny=20):
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
    meshgirds= np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
    return meshgirds

def draw_grid_byind(img,indx,indy,lshape=80):
    rectsize=640/lshape
    rct_ctx=(indx-0.5)*rectsize
    rct_cty=(indy-0.5)*rectsize
    cv2.rectangle(img,(int(rct_ctx),int(rct_cty)),(int(rct_ctx+rectsize),int(rct_cty+rectsize)), (0, 255, 0), 1)

def draw_gridanchor_byind(img,indx,indy,anchorsize,lshape=80):
    rectsize=640/lshape
    rct_ctx=(indx+0.5)*rectsize
    rct_cty=(indy+0.5)*rectsize
    w,h=tuple(anchorsize)
    # w*=rectsize
    # h*=rectsize

    # w*=8
    # h*=8

    # print('wh:',w,h)
    # cv2.rectangle(img,(int(rct_ctx),int(rct_cty)),(int(rct_ctx+rectsize),int(rct_cty+rectsize)), (0, 255, 0), 1)
    cv2.rectangle(img,(int(rct_ctx-w//2),int(rct_cty-h//2)),(int(rct_ctx+w//2),int(rct_cty+h//2)), (0, 255, 0), 1)

def limit_img_auto(imgin):
    img=np.array(imgin)
    sw = 1900 * 1.0
    sh = 1000 * 1.0
    h, w = tuple(list(imgin.shape)[0:2])
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    resize_ratio=sh/h
    if whratio > swhratio:
        resize_ratio=1.0*sw/w
    if resize_ratio<1:
        img=cv2.resize(imgin,None,fx=resize_ratio,fy=resize_ratio)
    return img


def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]

def label_xywh2xyxy(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
    return y

def get_ext_shape(pts_np):
    extrct=pts2rct(pts_np) 
    w=extrct[2]-extrct[0]
    h=extrct[3]-extrct[1]
    offsetx=-extrct[0]
    offsety=-extrct[1]
    return w,h,offsetx,offsety


def get_show_full(imgin,rctxyxy_list):
    img=np.array(imgin)
    imh,imw,imc=img.shape
    orih,oriw,oric=img.shape

    det_result=[]
    all_pts_np=[]
    # rctxyxy_list=np.array(rctxyxy_list)
    for rct in rctxyxy_list:
        det_result.append(rct)
        all_pts_np.append([rct[0],rct[1]])
        all_pts_np.append([rct[2],rct[3]])

    all_pts_np.extend([[0,0],[oriw,orih]])

    all_pts_np=np.array(all_pts_np)
    bdrct=pts2rct(all_pts_np)
    neww,newh,offsetx,offsety=get_ext_shape(all_pts_np)

    extimg=np.zeros((newh,neww,3),img.dtype)
    extimg[offsety:offsety+orih,offsetx:offsetx+oriw,:]=img.copy()

    for face_rectin in det_result:
        face_rect=np.array(face_rectin)
        # face_rect[::2]-=20
        face_rect[::2]+=offsetx
        face_rect[1::2]+=offsety

        # color=(0,255,255)
        color=(0,0,255)
        cv2.rectangle(extimg, (face_rect[0],face_rect[1]), (face_rect[2],face_rect[3]), color, 2)
        # for pt in face_land:
        #     cv2.circle(img, (int(pt[0]), int(pt[1])),int(3), (0, 0, 255), thickness=-1)
        #     all_pts_np.append((int(pt[0]), int(pt[1])))

    return extimg



if __name__=="__main__":
    # imroot=''
    path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\fortest'
    # path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie\refine'
    # path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie\refine\2'
    stride=32
    pad=0

    hyp_path='data/hyp.scratch.tiny.yaml'
    # Hyperparameters
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)  # load hyps


    dataset = LoadImagesAndLabelsDebug(path, 640, 1, augment=True, hyp=hyp,rect=False,cache_images=False,single_cls=True,
                                  stride=int(stride),pad=pad,image_weights=None,prefix='',tidl_load=False,kpt_label=5)  

    # torch.from_numpy(img), labels_out, self.img_files[index], shapes 

    self_anchors=np.array( 
    [   [4,5,  6,8,  10,12],  # P3/8
        [15,19,  23,30,  39,52],  # P4/16
        [72,97,  123,164,  209,297],  # P5/32
    ]
    )
    layer_sizes=[80,40,20]

    for i,items in enumerate(dataset):
        
        # if i!=8:
        #     continue
        print('i:',i)

        img,labels_out,impath,shapes=tuple(items)

        numpy_image = img.permute(1, 2, 0).cpu().numpy()#[:,:,::-1]

        #这种方式从tensor构造的image似乎不会响应cv2的绘制函数
        disimg=np.array(numpy_image)

        # disimg=cv2.resize(disimg,(640,640))

        disimg=cv2.cvtColor(numpy_image,cv2.COLOR_BGR2RGB)
        # disimg=cv2.resize(disimg,(640,640))#这种方式从tensor构造的image似乎不会响应cv2的绘制函数
        labels_out_np=labels_out.cpu().numpy()
        rctxyxy_list=[]
        rctlist=[]

        targets=[]
        for label in labels_out_np:
            rct=np.array(label[2:6])#*640
            # rct[0]+=0.4

            rctlist.append(rct)
            rctxyxy=xywh2xyxy_custom_p(rct)*640
            rctxyxy_list.append(np.array(rctxyxy,dtype=np.int32))

            targets.append(np.array(label[0:6]))#)
        targets=np.array(targets)
        targets=torch.tensor(targets)
        # print('targets:',targets.shape)

        ext_img=get_show_full(disimg,rctxyxy_list)

        for rctxyxy in rctxyxy_list:
            # print(disimg.shape,rct)
            cv2.rectangle(disimg,(rctxyxy[0],rctxyxy[1]),(rctxyxy[2],rctxyxy[3]), (0, 0, 255), 2)

            # 当mosaic中不加warp增强时
            # cv2.rectangle(disimg,(rctxyxy[0]*2,rctxyxy[1]*2),(rctxyxy[2]*2,rctxyxy[3]*2), (0, 0, 255), 2)

            # cv2.circle(disimg, (100,100),2, (0, 0, 255), thickness=-1)




        # cv2.circle(disimg, (100,100),10, (0, 0, 255), thickness=-1)
        cv2.imshow('disimg',limit_img_auto(disimg))
        cv2.imshow('ext_img',limit_img_auto(ext_img))
        if cv2.waitKey(0)==27:
            exit(0)


        # print(impath)
        # # print(shapes,labels_out)
        # # print(type(img),img.shape)
        # print(labels_out)
        # exit(0)

    # print(len(dataset))

    print('finish')