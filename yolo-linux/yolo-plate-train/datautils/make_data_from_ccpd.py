import cv2
import numpy as np
import  os,sys
import  numpy as np
import random
import torch
import math


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


def xyxy2xywh_custom(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[0][0] = x[:,0].mean()  # x center
    y[0][1] = x[:,1].mean()  # y center
    y[1][0] = x[1][0] - x[0][0]   # width
    y[1][1] = x[1][1] - x[0][1]  # height
    return y

def xywh2xyxy_custom(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
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

def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]


def find_nearest_point_index(x, y, pts_label_all):
    min_distance = float('inf')  # 初始化一个无限大的距离
    nearest_index = None

    for i, point in enumerate(pts_label_all):
        # 计算欧几里得距离
        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        
        # 如果当前点的距离小于最小距离，则更新最小距离和最近点的索引
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    return nearest_index

def rerank_plateland(faceland):
    extrct=pts2rct(faceland)
    plate_center=faceland.mean(axis=0).astype(np.int32)

    pts_bdrct=[
        [extrct[0],extrct[1]],
        [extrct[2],extrct[1]],
        plate_center,
        [extrct[2],extrct[3]],
        [extrct[0],extrct[3]],
    ]
    newland=np.array(faceland)
    for pt in faceland:
        ind=find_nearest_point_index(pt[0],pt[1],pts_bdrct)
        # find_nearest_point_index(x, y, pts_label_all):
        newland[ind]=pt

    newland=np.array(newland)
    return newland

def crop_pad(img,bdrct):
    h,w,c=img.shape

    extend_tlx = min(0,bdrct[0][0])
    extend_tly = min(0, bdrct[0][1])
    extend_brx = max(w,bdrct[1][0])
    extend_bry = max(h, bdrct[1][1])
    extendw=extend_brx-extend_tlx
    extendh = extend_bry - extend_tly
    bkimg=np.zeros((extendh,extendw,3),img.dtype)
    # print('cropimg',extend_tlx,extend_tly)

    xshift=0-extend_tlx
    yshift = 0 - extend_tly
    bkimg[yshift:yshift+h,xshift:xshift+w]=img

    bdrct[:,0]+=xshift
    bdrct[:, 1] += yshift

    cropimg=bkimg[bdrct[0][1]:bdrct[1][1],bdrct[0][0]:bdrct[1][0]]
    return cropimg,xshift,yshift

def bigger_rct(wratio,hratio,rct):
    wratio=(wratio-1.0)*0.5
    hratio=(hratio-1.0)*0.5
    delta_w=(rct[2])*wratio+0.5
    delta_h=(rct[3])*hratio+0.5
    # return limit_rct([int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)],imgshape)
    return [int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)]

def pred_plateland(imgfull,ptsin,plateland_net):
    cropw=256
    croph=144

    bdrct=pts2rct(ptsin)
    bdrct = [bdrct[0], bdrct[1], bdrct[2] - bdrct[0], bdrct[3] - bdrct[1]]
    bdrct = bigger_rct(1.5, 1.6, bdrct)
    bdrct = [bdrct[0], bdrct[1], bdrct[0] + bdrct[2], bdrct[1] + bdrct[3]]
    # cv2.rectangle(img, (bdrct[0], bdrct[1]), (bdrct[2], bdrct[3]), (255, 0, 0), 4)
    curbdrct = bdrct
    curbdrct = np.array([[curbdrct[0], curbdrct[1]], [curbdrct[2], curbdrct[3]]])
    cropimg, xshift, yshift = crop_pad(imgfull, curbdrct)
    h, w, c = cropimg.shape
    hratio = 1.0 * croph / h
    wratio = 1.0 * cropw / w
    cropimg = cv2.resize(cropimg, (cropw, croph))

    cropimgori = np.array(cropimg)
    cropimg = cropimg.astype(np.float32) / 255.0

    # img=cv2.resize(img,(128,128))

    cropimg = cropimg.transpose(2, 0, 1)
    cropimg = torch.from_numpy(cropimg).unsqueeze(0)
    cropimg = cropimg.to('cuda')
    # out=net(img).cpu().numpy()
    land_pred = plateland_net(cropimg)
    land_pred = land_pred.cpu().detach().numpy().reshape(8)
    # print(land_pred.shape, land_pred)

    quadpred = land_pred.astype(np.float32)
    quadpred[::2] *= cropw* 1.0
    quadpred[1::2] *= croph * 1.0

    quadpred = quadpred.astype(np.float32).reshape((4,2))
    # for k in range(0,4):
    #     cv2.circle(cropimgori, tuple(np.array(quadpred[k],dtype=np.int32)), 4,(0, 0, 0), thickness=-1)

    quadpred[:,0]/=wratio
    quadpred[:,1]/=hratio
    quadpred[:,0]+=xshift+bdrct[0]
    quadpred[:,1]+=yshift+bdrct[1]

    quadpred = quadpred.astype(np.int32)

    # for k in range(0,4):
    #     cv2.circle(imgfull, tuple(quadpred[k]), 10,(0, 0, 0), thickness=-1)
    # cv2.imshow('imgfull',imgfull)
    # cv2.imshow('cropimgori',cropimgori)


    return quadpred

def parse_label(label_flatten,img):
    h,w,c=img.shape

    rct_list=[]
    land_list=[]

    for label_flatten in label_flatten_list:

        label_flatten[1:] = xywhn2xyxy(label_flatten[1:], w,h, padw=0, padh=0,kpt_label=True)
        xyxy = np.array(label_flatten[1:5]).astype(np.int32)
        kptsx = label_flatten[5::2].astype(np.int32)
        kptsy = label_flatten[6::2].astype(np.int32)

        rct_list.append([[xyxy[0],xyxy[1]], [xyxy[2],xyxy[3]]])
        plate_land=[]
        for k in range(0,5):
            plate_land.append([kptsx[k],kptsy[k]])
        land_list.append(plate_land)

        # cv2.rectangle(img, [xyxy[0],xyxy[1]], [xyxy[2],xyxy[3]], (0, 255, 0), thickness=2)

        # for k in range(0,5):
        #     cv2.circle(img, (kptsx[k],kptsy[k]), 2,(0, 255, 255), thickness=-1)
    rct_list=list(np.array(rct_list,dtype=np.int32))
    land_list=list(np.array(land_list,dtype=np.int32))

    return rct_list,land_list

# 生成裁剪框，将车牌裁剪出来作为新的数据，最好是裁剪成正方形，这样就能维持车牌在图片中的占比
def get_aug_palte_rct(img,platerct,plateland):
    h,w,c=img.shape
    platerct=np.array(platerct,dtype=np.float32)
    plateland=np.array(plateland,dtype=np.float32)
    plate_center=platerct.mean(axis=0).astype(np.int32)
    # print('plate_center:',plate_center)
    platew=platerct[1][0]-platerct[0][0]
    plateh=platerct[1][1]-platerct[0][1]
    maxsize=max(platew,plateh)

    # ratio_max=0.95
    # rato_min=0.7

    # ratio=0.4+0.55*random.random()
    # ratio=0.9+0.05*random.random()
    ratio=0.05+0.9*random.random()
    # ratio=0.05
    halfnewsize=maxsize/ratio*0.5
    croprct=[plate_center[0]-halfnewsize,plate_center[1]-halfnewsize,plate_center[0]+halfnewsize,plate_center[1]+halfnewsize]
    croprct=np.array(croprct)

    #越界情况下更新图片和label 
    newrct=np.array(platerct)
    newland=np.array(plateland)
    full_bdpts=np.array([[croprct[0],croprct[1]],[croprct[2],croprct[3]],[0,0],[w,h]])
    fullbdrct=pts2rct(full_bdpts)
    newrct[:,0 ]-=fullbdrct[0]
    newrct[:,1]-=fullbdrct[1]
    newland[:,0]-=fullbdrct[0]
    newland[:,1]-=fullbdrct[1]
    fullbdrct=np.array(fullbdrct,dtype=np.int32)
    # print('-fullbdrct[1], fullbdrct[3]-h, -fullbdrct[0], fullbdrct[2]-w:',-fullbdrct[1], fullbdrct[3]-h, -fullbdrct[0], fullbdrct[2]-w)
    fullimg=cv2.copyMakeBorder(img, -fullbdrct[1], fullbdrct[3]-h, -fullbdrct[0], fullbdrct[2]-w, cv2.BORDER_CONSTANT, None, (0,0,0))
    newrct=newrct.astype(np.int32)
    

    # 越界图片更新了，裁剪位置也要更新
    croprct[::2]-=fullbdrct[0]
    croprct[1::2]-=fullbdrct[1]

    croprct=np.array(croprct,dtype=np.int32)
    # cv2.rectangle(fullimg, tuple(newrct[0]), tuple(newrct[1]), (0, 255, 0), thickness=2)
    # cv2.rectangle(fullimg, (int(croprct[0]),int(croprct[1])), (int(croprct[2]),int(croprct[3])), (0, 255, 0), thickness=2)

    # cv2.imshow('fullimg',limit_img_auto(fullimg))

    croped_img=fullimg[croprct[1]:croprct[3],croprct[0]:croprct[2],:]
    # cv2.imshow('croped_img',limit_img_auto(croped_img))


    newrct[:,0 ]-=croprct[0]
    newrct[:,1]-=croprct[1]
    newland[:,0]-=croprct[0]
    newland[:,1]-=croprct[1]

    # newrct[:,0]-=croprct[0]
    # newrct[:,1]-=croprct[1]

    # newland[:,0]-=croprct[0]
    # newland[:,1]-=croprct[1]

    # return newrct,newland
    targetsize=1024
    h,w,c=croped_img.shape
    croped_img=cv2.resize(croped_img,(targetsize,targetsize))
    
    scale_val=targetsize/w*1.0
    newrct=newrct.astype(np.float32)
    newland=newland.astype(np.float32)
    newrct*=scale_val
    newland*=scale_val

    return croped_img,newrct,newland


def makedir(filedir):
    if os.path.exists(filedir) is False:
        os.mkdir(filedir)


def make_facenao_lines(oriimg,face_label_list):

    h,w,c=oriimg.shape
    lines=''
    for face_label in face_label_list:
        line='0 '
        rect_label,faceland_label=face_label

        rect_label_xywh=np.array(rect_label)
        faceland_label=np.array(faceland_label)


        rect_label_xywh=xyxy2xywh_custom(rect_label_xywh)
        for pt in rect_label_xywh:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' '

        for pt in faceland_label:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' 2.0 '
        line=line.rstrip(' ')+'\n'
        lines+=line
    lines=lines.rstrip('\n')
    return lines

if __name__=='__main__':
    # dataroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch01'
    # txtroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/labels_yolov7/ccpd'

    # dataroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataTest1k'
    # txtroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/labels_yolov7/test1k'

    # dataroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch01/images'
    # txtroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch01/labels'

    dataroot=r'/disks/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch01/images'
    txtroot=r'/disks/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch01/labels'

    dstroot=r'/disks/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch01-scaleaug'
    dstroot_image=os.path.join(dstroot,'images')
    dstroot_label=os.path.join(dstroot,'labels')
    makedir(dstroot_image)
    makedir(dstroot_label)


    plate_land_pt=r'/home/tao/disk1/Workspace/Project/Pytorch/TSLPR/plate_landmark/plateland.pt'
    plateland_net=torch.jit.load(plate_land_pt)

    torch.set_grad_enabled(False)
    ims=get_ims(dataroot)

    

    # ims.sort()
    # ims.sort(reverse=True)
    random.shuffle(ims)

    

    colors = [(0, 0, 255),  # 红色 (B, G, R)
            (0, 165, 255),  # 橙色
            (0, 255, 255),  # 黄色
            (0, 255, 0),  # 绿色
            (255, 0, 0),  # 蓝色
            (255, 255, 0),  # 青色
            (128, 0, 128)]  # 紫色

    
    ims=ims[0:50000]
    # ims=ims[0:5]


    numims=len(ims)
    for ind,impath in enumerate(ims):
        print('{}of{}'.format(ind,numims))
        
        imname=os.path.basename(impath)
        imkey,ext=split_imkey(impath)
        txtname=imkey+'.txt'
        txtpath=os.path.join(txtroot,txtname)

        img_const = cv2.imread(impath)
        h,w,c=img_const.shape
        img=np.array(img_const)

        # print(impath,txtpath)
        label_flatten_list=load_yololabel(txtpath)

        rct_list,land_list=parse_label(label_flatten_list,img_const)

        croped_img,newrct,newland=get_aug_palte_rct(img,rct_list[0],land_list[0])
        newland=np.array(newland,dtype=np.int32)
        newrct=np.array(newrct,dtype=np.int32)



        # cv2.rectangle(croped_img ,newrct[0], newrct[1], (0, 255, 0), thickness=2)
        # for k in range(0,5):
        #     cv2.circle(croped_img, (newland[k][0],newland[k][1]), 2,(0, 255, 255), thickness=-1)
        # cv2.imshow('croped_img',limit_img_auto(croped_img))
        # for i,rct in enumerate(rct_list):
        #     cv2.rectangle(img, rct[0], rct[1], (0, 255, 0), thickness=2)

        # print('newrct:',newrct)

        face_label_list=[(newrct,newland)]
        anolines=make_facenao_lines(croped_img,face_label_list)
        # print('anolines:',anolines)

        dstimgpath=os.path.join(dstroot_image,imname)
        dstlabelpath=os.path.join(dstroot_label,imkey+'.txt')

        cv2.imwrite(dstimgpath,croped_img)
        with open(dstlabelpath,'w') as f:
            f.writelines(anolines)

        
        # # for k in range(0,5):
        # #     cv2.circle(img, (kptsx[k],kptsy[k]), 2,(0, 255, 255), thickness=-1)
        # # img, ratio, (dw, dh) = letterbox(img_const, 640, stride=32)
        # cv2.imshow('img', limit_img_auto(img))
        # print('img.shape:',img.shape)
        # if cv2.waitKey(0)==27:
        #     exit(0)

    print('finish')