from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform
import math
import torch
############路径及窗口大小设置############################
# local_path = u"Z:\workspace\yoloface\dataset\crawler"          #图像根目录
# local_path=r'Z:\workspace\yoloface\dataset\label_test'
# local_path=r'Z:\workspace\yoloface\dataset\crawler'
# local_path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000'
# local_path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie\refine\2'
# local_path=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie_test'
local_path=[
    # r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie_test'
    # r'Z:\workspace\yoloface\dataset\hair_zhedang_label'
    # r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00'
    # r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/labelplate_test'
    r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00'
    # r'/home/tao/platedata/plate/refine'
][0]


MAX_WIDTH=1800                    #图像窗口最大宽度
MAX_HEIGHT=1000                    #图像窗口最大高度
########################################

global_facerctpts_label=[]
global_facelandpts_label=[]
global_face_label_list=[]
global_flags=[0,0,0,0,0] 

global_extend_lenth_list=[0,0,0,0]
global_img_info=[0,0,0,0] #img,h,w,c
global_offset=[0,0]


# global_extra_offset=[0,0]
# global_prev_stats=[-1,-1,-1,-1]
#cursor_rect,cursor_land,curstate,
# 方框的光标，关键点的光标，当前窗口处于哪个阶段。
# global_face_label_list_origin=[]
# global_edge_center_list=[[0,0],[0,0],[0,0],[0,0]]




img=None


local_path=local_path+'/'
key_dic={}
def load_key_val():
    key_val_path='key_val.txt'
    if 'Windows' in platform.system():
        key_val_path='key_val_win.txt'
    lines=open(key_val_path).readlines()
    for line in lines:
        item=line.split(' ')
        vals=item[1].split(',')
        val_lst=[]
        for val in vals:
            val_lst.append(int(val))
        key_dic[item[0]]=val_lst
        # print item[0],val_lst
load_key_val()

def limit_imgw(img):
    if  img.shape[1]>360:
        img=cv2.resize( img,(360,int(360.0/img.shape[1]*img.shape[0])),interpolation=cv2.INTER_CUBIC )
    if  img.shape[0]>360:
        img=cv2.resize( img,(int(360.0/img.shape[0]*img.shape[1]),360),interpolation=cv2.INTER_CUBIC )
    return  img


def cac_winratio(disimg,_MAX_WIDTH,_MAX_HEIGHT ):
    wm_ratio=1.0
    if disimg.shape[1] > _MAX_WIDTH or disimg.shape[0] > _MAX_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (_MAX_WIDTH / float(_MAX_HEIGHT)):
            # cv2.resizeWindow(winname, MAX_WIDTH, int(MAX_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = _MAX_WIDTH / float(disimg.shape[1])
        else:
            # cv2.resizeWindow(winname, int(MAX_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), MAX_HEIGHT)
            wm_ratio = _MAX_HEIGHT / float(disimg.shape[0])
    
    return wm_ratio

def cac_winratio_force(disimg,_MAX_WIDTH,_MAX_HEIGHT ):
    wm_ratio=1.0
    # if disimg.shape[1] > _MAX_WIDTH or disimg.shape[0] > _MAX_HEIGHT:
    if (disimg.shape[1] / float(disimg.shape[0])) > (_MAX_WIDTH / float(_MAX_HEIGHT)):
        # cv2.resizeWindow(winname, MAX_WIDTH, int(MAX_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
        wm_ratio = _MAX_WIDTH / float(disimg.shape[1])
    else:
        # cv2.resizeWindow(winname, int(MAX_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), MAX_HEIGHT)
        wm_ratio = _MAX_HEIGHT / float(disimg.shape[0])
    
    return wm_ratio


def limit_window(disimg,winname):
    wm_ratio=1.0
    if disimg.shape[1] > MAX_WIDTH or disimg.shape[0] > MAX_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (MAX_WIDTH / float(MAX_HEIGHT)):
            cv2.resizeWindow(winname, MAX_WIDTH, int(MAX_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = MAX_WIDTH / float(disimg.shape[1])
        else:
            cv2.resizeWindow(winname, int(MAX_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), MAX_HEIGHT)
            wm_ratio = MAX_HEIGHT / float(disimg.shape[0])
    else:
        cv2.resizeWindow(winname, disimg.shape[1], disimg.shape[0])
    return wm_ratio

def resize_facecrop_window(disimg,winname):
    wm_ratio=1.0
    TARGET_WIDTH=800
    TARGET_HEIGHT=800
    # if disimg.shape[1] > TARGET_WIDTH or disimg.shape[0] > TARGET_HEIGHT:
    if (disimg.shape[1] / float(disimg.shape[0])) > (TARGET_WIDTH / float(TARGET_HEIGHT)):
        cv2.resizeWindow(winname, TARGET_WIDTH, int(TARGET_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
        wm_ratio = TARGET_WIDTH / float(disimg.shape[1])
    else:
        cv2.resizeWindow(winname, int(TARGET_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), TARGET_HEIGHT)
        wm_ratio = TARGET_HEIGHT / float(disimg.shape[0])
    # else:
    #     cv2.resizeWindow(winname, disimg.shape[1], disimg.shape[0])
    return wm_ratio


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def get_dirims(imroot):
    imgpathlst=[]

    filenames=os.listdir(imroot)
    for filename in filenames:
        if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
            imgpathlst.append(os.path.join(imroot,filename))
    return imgpathlst

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


###########################
def get_facecrop_rect(img,pt1,pt2):
    h,w,c=img.shape
    pts2=np.array([np.array(pt1),np.array(pt2)])
    pt_ct=pts2.mean(axis=0)
    pts_res=pts2-pt_ct
    pts_res[:,0]*=1.5 #x
    pts_res[:,1]*=1.5 #y
    pts_res+=pt_ct
    # print(pts2)
    # print(pt_ct)
    pts_res=pts_res.astype(np.int32)
    pts_res[:,0]=np.clip(pts_res[:,0],0,w-1)
    pts_res[:,1]=np.clip(pts_res[:,1],0,h-1)
    return pts_res


# def find_nearest_rct_index(x, y, face_label_list):
#     min_distance = float('inf')  # 初始化一个无限大的距离
#     nearest_index = 0

#     for i,face_label in enumerate(face_label_list):
#         rect_label,faceland_label=face_label
#         distance =0
#         for point in rect_label:
#             distance += math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
#         if distance < min_distance:
#             min_distance = distance
#             nearest_index = i

#     return nearest_index


def find_nearest_rct_index(x, y, face_label_list):
    min_distance = float('inf')  # 初始化一个无限大的距离
    nearest_index = 0

    for i,face_label in enumerate(face_label_list):
        rect_label,faceland_label=face_label
        distance =0

        rect_label=np.array(rect_label)
        rect_label=rect_label.mean(axis=0)
        distance=math.sqrt((x - rect_label[0]) ** 2 + (y -rect_label[1]) ** 2)

        # for point in rect_label:
        #     distance += math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    return nearest_index


def draw_cross(image,pt,line_color,line_length,line_thickness):
    
    center_x, center_y=pt
    # 绘制水平线
    cv2.line(image, (center_x - line_length, center_y), (center_x + line_length, center_y), line_color, line_thickness)
    # 绘制垂直线
    cv2.line(image, (center_x, center_y - line_length), (center_x, center_y + line_length), line_color, line_thickness)



def update_view_main():

    rectline_thick_base=2.0
    circle_r_base=3.0
    circle_r_offset=2.0
    # rectline_thick_base

    disimg=np.array(img)
    linethick_scale_ratio=cac_winratio(disimg,MAX_WIDTH,MAX_HEIGHT)
    # print('linethick_scale_ratio:',linethick_scale_ratio)
    line_ratio_mul=1/linethick_scale_ratio


    h,w,c=disimg.shape


    edge_center_list=[np.array([w//2,0]),np.array([w,h//2]),np.array([w//2,h]),np.array([0,h//2])]
    cross_cursor_ind=global_flags[3]
    for i,pt in enumerate(edge_center_list):
        draw_cross(disimg,tuple(pt),(0, 0, 255),int(2*circle_r_offset+circle_r_base*line_ratio_mul+0.5),int(2*rectline_thick_base*line_ratio_mul+0.5))
    draw_cross(disimg,tuple(edge_center_list[cross_cursor_ind]),(0, 255, 0),int(2*circle_r_offset+circle_r_base*line_ratio_mul+0.5),int(2*rectline_thick_base*line_ratio_mul+0.5))


    for pt in global_facerctpts_label:
        cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), (0, 0, 255), thickness=-1)

    if len(global_facerctpts_label)==2:
        # rect_cursor_ind=global_flags[0]
        # pt_rect_cur=global_facerctpts_label[rect_cursor_ind]
        # cv2.circle(disimg, (pt_rect_cur[0], pt_rect_cur[1]),int(3), (0, 0, 255), thickness=-1)

        tl=global_facerctpts_label[0]
        br=global_facerctpts_label[1]   
        cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), int(rectline_thick_base*line_ratio_mul+0.5))
        
        bigrct=get_facecrop_rect(disimg,tl,br)
        tl=bigrct[0]
        br=bigrct[1] 
        # cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 255, 255), 1)

    if len(global_face_label_list)>0:
        for face_label in global_face_label_list:
            rect_label,faceland_label=face_label
            for pt in faceland_label:
                cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), (0, 0, 255), thickness=-1)

            tl=rect_label[0]
            br=rect_label[1]   
            cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), int(rectline_thick_base*line_ratio_mul+0.5))

        rect_cursor=global_flags[0]
        # print(rect_cursor)

        rect_label,faceland_label=global_face_label_list[rect_cursor]
        tl,br=rect_label[0],rect_label[1]  
        cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 255, 0), int(rectline_thick_base*line_ratio_mul+0.5))

        for pt in faceland_label:
            cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), (0, 255, 0), thickness=-1)


    cv2.imshow('img', disimg)
    limit_window(disimg,'img')


def update_view_facecrop():
    global global_facecroped
    # tl=global_facerctpts_label[0]
    # br=global_facerctpts_label[1]  
    # croprct=get_facecrop_rect(img,tl,br)
    # tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
    # facecroped=img[tly:bry+1,tlx:brx+1,:].copy()

    colors = [(0, 0, 255),  # 红色 (B, G, R)
            (0, 165, 255),  # 橙色
            (0, 255, 255),  # 黄色
            (0, 255, 0),  # 绿色
            (255, 0, 0),  # 蓝色
            (255, 255, 0),  # 青色
            (128, 0, 128)]  # 紫色

    disimg=np.array(global_facecroped)


    TARGET_WIDTH=800
    TARGET_HEIGHT=800
    rectline_thick_base=1.5
    circle_r_base=4.0
    circle_r_offset=3.0
    string_base=1.0
    fontbase=0.4

    # rectline_thick_base
    linethick_scale_ratio=cac_winratio_force(disimg,TARGET_WIDTH,TARGET_HEIGHT)
    # print('linethick_scale_ratio:',linethick_scale_ratio)
    line_ratio_mul=1/linethick_scale_ratio

    tl=global_facerctpts_label[0]
    br=global_facerctpts_label[1]   
    cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), int(rectline_thick_base*line_ratio_mul+0.5))
    for pt in global_facerctpts_label:
        cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), (0, 0, 255), thickness=-1)


    for i,pt in enumerate(global_facelandpts_label):
        # cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_base*line_ratio_mul+0.5), (0, 0, 255), thickness=-1)
        cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), colors[i], thickness=-1)
        # cv2.putText(disimg, str(i), (pt[0], pt[1]),cv2.FONT_HERSHEY_SIMPLEX, fontbase*line_ratio_mul, (0, 0, 255), int(string_base*line_ratio_mul+0.5), cv2.LINE_AA)

    pt_cursor=global_flags[1]
    if len(global_facelandpts_label)==5:
        if pt_cursor>=2:
            pt=global_facelandpts_label[pt_cursor-2]
            # cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_base*line_ratio_mul+0.5), (0, 255, 0), thickness=-1)
            cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+1.5*circle_r_base*line_ratio_mul+0.5), colors[pt_cursor-2], thickness=-1)
            cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+1.5*circle_r_base*line_ratio_mul+0.5), colors[-1], thickness=int(1.5*line_ratio_mul+0.5))
            # cv2.putText(disimg, str(pt_cursor-2), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, fontbase*line_ratio_mul, (0, 255, 0), int(string_base*line_ratio_mul+0.5), cv2.LINE_AA)
        else:
            pt=global_facerctpts_label[pt_cursor]
            cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+1.5*circle_r_base*line_ratio_mul+0.5), (0, 255, 0), thickness=-1)
            


    cv2.imshow('refine_face',disimg)
    resize_facecrop_window(disimg,'refine_face')

def find_closest_point(edge_center_list, pt):
    # 计算每个点与 pt 之间的距离
    distances = np.linalg.norm(edge_center_list - pt, axis=1)
    # 找到最小距离的索引
    closest_index = np.argmin(distances)
    return closest_index


def mouse_func_facerect(event,x,y,flags,param):
    global global_facerctpts_label
    global global_facelandpts_label
    global global_face_label_list

    # if len(global_facerctpts_label) == 2:
    #     return
    # global_flags
    if global_flags[2]!=0:
        return
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(global_facerctpts_label) < 2:
            # print(x,y)
            global_facerctpts_label.append([x,y])
            # update_view_main()
        if len(global_facerctpts_label)==2:
            # get4pts()
            # get_info()
            update_view_main()
            refine_face()

        update_view_main()
        cv2.waitKey(1)#注意此处等待按键



    if event == cv2.EVENT_RBUTTONDOWN:
        # print('cv2.EVENT_RBUTTONDOWN')
        if len(global_face_label_list)>0:
            rect_cursor=global_flags[0]
            rect_label,faceland_label=global_face_label_list[rect_cursor]

            tl,br=rect_label[0],rect_label[1]  
            croprct=get_facecrop_rect(img,tl,br)
            tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
            faceland_label=np.array(faceland_label)
            faceland_label[:,0]-=tlx
            faceland_label[:,1]-=tly
            # print(faceland_label)

            global_facerctpts_label=list(rect_label)
            # print('global_facerctpts_label:',global_facerctpts_label)
            global_facelandpts_label=list(faceland_label)
            del global_face_label_list[rect_cursor]
            global_flags[0]=0
            refine_face()

    if event == cv2.EVENT_MOUSEMOVE:

        edge_center_list=np.array([np.array([w//2,0]),np.array([w,h//2]),np.array([w//2,h]),np.array([0,h//2])])
        cross_cursor_ind=find_closest_point(edge_center_list, [x, y])
        global_flags[3]=cross_cursor_ind
        global_flags[0]=find_nearest_rct_index(x, y, global_face_label_list)
        update_view_main()
        
        # if global_prev_stats[0]!=global_flags[0]:
        #     update_view_main()
        #     global_prev_stats[0]=global_flags[0]
        # cv2.waitKey(1)#注意此处等待按键

        

def mouse_func_faceland(event,x,y,flags,param):
    global global_facecroped

    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(global_facelandpts_label) < 5:
            # print(x,y)
            global_facelandpts_label.append([x,y])
            # update_view_main()
        elif len(global_facelandpts_label)==5:
            # get4pts()
            # get_info()
            # refine_face()
            pt_cursor=global_flags[1]
            if pt_cursor>=2:
                global_facelandpts_label[pt_cursor-2]=[x,y]
            else:
                global_facerctpts_label[pt_cursor]=[x,y]
            pass

        update_view_facecrop()
        cv2.waitKey(1)#注意此处等待按键



    if event == cv2.EVENT_MOUSEMOVE:
        # 注释掉，人脸框点位的选取
        # pts_label_all=list(global_facerctpts_label)
        # 强行置为很远的点
        pts_label_all=[[10000,10000],[10000,10000]]
        pts_label_all.extend(global_facelandpts_label)
        # print(len(pts_label_all))
        # pts_label_all=np.array(pts_label_all)
        global_flags[1]=find_nearest_point_index(x, y, pts_label_all)
        
        update_view_facecrop()
        # cv2.waitKey(1)#注意此处等待按键

    # if event == cv2.EVENT_RBUTTONDOWN:
    #     global_flags[4]=1

# 

# global_facelandpts_label
def refine_face():
    global global_facecroped
    global global_facerctpts_label

    global_flags[2]=1

    # cropimg=np.array(img)
    tl=global_facerctpts_label[0]
    br=global_facerctpts_label[1]  
    croprct=get_facecrop_rect(img,tl,br)
    tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
    global_facecroped=img[tly:bry+1,tlx:brx+1,:].copy()


    facerctpts_label=np.array(global_facerctpts_label)
    facerctpts_label[:,0]-=tlx
    facerctpts_label[:,1]-=tly
    global_facerctpts_label=list(facerctpts_label)


    cv2.namedWindow('refine_face', cv2.WINDOW_FREERATIO)
    # cv2.imshow('refine_face',global_facecroped)
    update_view_facecrop()
    cv2.setMouseCallback('refine_face', mouse_func_faceland)

    # update_view_main()
    breakflag=0
    while 1:
        key=cv2.waitKey(0)
        # if key in key_dic['SPACE']:
        #     global_flags[0]=1-global_flags[0]
        if key in key_dic['ENTER']:
            facelandpts_label_tofull=np.array(global_facelandpts_label)
            facelandpts_label_tofull[:,0]+=tlx
            facelandpts_label_tofull[:,1]+=tly

            platerct_from_land=pts2rct(facelandpts_label_tofull)
            facerctpts_label_tofull=np.array(platerct_from_land).reshape((2,2))
            global_facerctpts_label=list(facerctpts_label_tofull)


            # facerctpts_label_tofull=np.array(facerctpts_label_tofull)
            # facerctpts_label_tofull[:,0]+=tlx
            # facerctpts_label_tofull[:,1]+=tly
            ###### global_facerctpts_label=list(facerctpts_label_tofull)

            global_face_label_list.append((list(facerctpts_label_tofull),list(facelandpts_label_tofull)))

            cv2.destroyWindow('refine_face')
            global_facelandpts_label.clear()
            global_facerctpts_label.clear()
            global_flags[2]=0
            breakflag=1
            update_view_main()
            break
        if key in key_dic['BACK']:
            if len(global_facelandpts_label)>0:
                global_facelandpts_label.pop()

        if breakflag!=1:
            # update_view_main()
            update_view_facecrop()

def parse_path(filepath):
    filename=os.path.basename(filepath)
    ext='.'+filename.split('.')[-1]
    path_noext=filepath[0:len(filepath)-len(ext)]
    filekey=filename[0:len(filename)-len(ext)]
    return filekey,ext,path_noext

def imgpath_to_txtpath(imgpath):
    filekey,ext,path_noext=parse_path(imgpath)
    txtpath=path_noext+'.txt'
    return txtpath

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[0][0] = x[:,0].mean()  # x center
    y[0][1] = x[:,1].mean()  # y center
    y[1][0] = x[1][0] - x[0][0]   # width
    y[1][1] = x[1][1] - x[0][1]  # height
    return y

def xywh2xyxy(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
    return y


def make_facenao_lines():
    oriimg,oh,ow,oc=tuple(global_img_info)
    offsetx=global_offset[0]
    offsety=global_offset[1]

    h,w,c=oriimg.shape
    lines=''
    for face_label in global_face_label_list:
        line='0 '
        rect_label,faceland_label=face_label

        rect_label_xywh=np.array(rect_label)
        faceland_label=np.array(faceland_label)
        # 调整偏移量
        rect_label_xywh[:,0]-=offsetx
        rect_label_xywh[:,1]-=offsety
        faceland_label[:,0]-=offsetx
        faceland_label[:,1]-=offsety

        rect_label_xywh=xyxy2xywh(rect_label_xywh)
        for pt in rect_label_xywh:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' '

        for pt in faceland_label:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' 2.0 '
        line=line.rstrip(' ')+'\n'
        lines+=line
    lines=lines.rstrip('\n')
    return lines


def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]


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


def load_ano_from_txt(txtpath,w,h):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    face_label_list_loaded=[]


    ptsall=[]
    for line in lines:
        numbers = line.split(' ')
        num_array = np.array([float(number) for number in numbers])
        pts_array = np.delete(num_array[5:], np.arange(2, num_array.shape[0] - 5,3))  # remove the occlusion paramater from the GT
        # num_array = np.hstack((num_array[:5], pts_array))
        
        facerect=num_array[1:5].reshape(2,2)
        faceland=pts_array.reshape(-1,2)

        # print('facerect[0][0]:',facerect[0][0])
        # facerect[0][0]+=0.18
        # print('facerect[0][0]:',facerect[0][0])
        facerect=xywh2xyxy(facerect)
        # faceland=xywh2xyxy(faceland)

        ptsoneface = np.concatenate((facerect, faceland), axis=0)

        facerect[:,0]*=w
        facerect[:,1]*=h
        faceland[:,0]*=w
        faceland[:,1]*=h

        faceland=rerank_plateland(faceland)

        # print(facerect)

        face_label_list_loaded.append((list(facerect.astype(np.int32)),list(faceland.astype(np.int32))))

        ptsall.append(ptsoneface)
    
    if len(ptsall)>0:
        ptsall=np.concatenate(tuple(ptsall),axis=0)
        extrct=pts2rct(ptsall)
            # break
        # print('extrct:',extrct)

    return face_label_list_loaded

def makedir(filedir):
    if os.path.exists(filedir) is False:
        os.mkdir(filedir)


def update_extend():
    if len(global_face_label_list)<=0:
        return 
    offsetxf,offsetyf=0,0
    global img
    ptsall=[]
    oriimg,oh,ow,oc=tuple(global_img_info)
    if len(global_face_label_list)>0:
        for face_label in global_face_label_list:
            rect_label,faceland_label=face_label
            rect_label_np=np.array(rect_label,np.float32)
            faceland_label_np=np.array(faceland_label,np.float32)
            rect_label_np[:,0]/=ow
            rect_label_np[:,1]/=oh
            faceland_label_np[:,0]/=ow
            faceland_label_np[:,1]/=oh
            ptsall.append(np.concatenate((rect_label_np,faceland_label_np),axis=0))

    offsetxf=global_extend_lenth_list[0]
    offsetyf=global_extend_lenth_list[1]

    extw=int((1+global_extend_lenth_list[0]+global_extend_lenth_list[2])*w)
    exth=int((1+global_extend_lenth_list[1]+global_extend_lenth_list[3])*h)

    newimg=np.zeros((exth,extw,3),img.dtype)
    # print(offsety,offsety+h,offsetx,offsetx+w)
    # exit(0)
    offsety=int(oh*offsetyf)
    offsetx=int(ow*offsetxf)



    newimg[offsety:offsety+h,offsetx:offsetx+w,:]=oriimg.copy()
    img=newimg
    
    if len(global_face_label_list)>0:
        for i,face_label in enumerate(global_face_label_list):
            rect_label,faceland_label=face_label
            rect_label=np.array(rect_label)
            faceland_label=np.array(faceland_label)
            rect_label[:,0]+=offsetx-global_offset[0]
            rect_label[:,1]+=offsety-global_offset[1]

            faceland_label[:,0]+=offsetx-global_offset[0]
            faceland_label[:,1]+=offsety-global_offset[1]
            global_face_label_list[i]=(list(rect_label),list(faceland_label))
    global_offset[0]=offsetx
    global_offset[1]=offsety


def update_extend_load():
    if len(global_face_label_list)<=0:
        return 
    
    offsetxf,offsetyf=0,0

    global img
    ptsall=[]
    oriimg,oh,ow,oc=tuple(global_img_info)
    if len(global_face_label_list)>0:
        for face_label in global_face_label_list:
            rect_label,faceland_label=face_label
            rect_label_np=np.array(rect_label,np.float32)
            faceland_label_np=np.array(faceland_label,np.float32)

            rect_label_np[:,0]/=ow
            rect_label_np[:,1]/=oh

            faceland_label_np[:,0]/=ow
            faceland_label_np[:,1]/=oh

            ptsall.append(np.concatenate((rect_label_np,faceland_label_np),axis=0))

            print('rect_label:',rect_label)
    ptsall=np.concatenate(tuple(ptsall),axis=0)
    prct=pts2rct(ptsall)
    print(prct)


    ptall=np.array([prct[0],prct[1],prct[2],prct[3]])
    ptall[0:2]=np.maximum(-ptall[0:2],0)
    ptall[2:4]=np.maximum(ptall[2:4]-1,0)
    ptmask=np.array(ptall>0,dtype=np.float32)
    ptall=0.1*(np.array(ptall*10,dtype=np.int32)+1)*ptmask
    global_extend_lenth_list[0],global_extend_lenth_list[1],global_extend_lenth_list[2],global_extend_lenth_list[3]=ptall[0],ptall[1],ptall[2],ptall[3]


    # if prct[0]<0:
    #     global_extend_lenth_list[0]=(int(-prct[0]*10)+1)*0.1
    #     offsetxf=global_extend_lenth_list[0]
    offsetxf=global_extend_lenth_list[0]
    offsetyf=global_extend_lenth_list[1]


    print('global_extend_lenth_list:',global_extend_lenth_list)

    extw=int((1+global_extend_lenth_list[0]+global_extend_lenth_list[2])*w)
    exth=int((1+global_extend_lenth_list[1]+global_extend_lenth_list[3])*h)

    newimg=np.zeros((exth,extw,3),img.dtype)
    # print(offsety,offsety+h,offsetx,offsetx+w)
    # exit(0)
    offsety=int(oh*offsetyf)
    offsetx=int(ow*offsetxf)

    newimg[offsety:offsety+h,offsetx:offsetx+w,:]=oriimg.copy()
    img=newimg

    # print('global_offset:',global_offset)

    
    if len(global_face_label_list)>0:
        for i,face_label in enumerate(global_face_label_list):
            rect_label,faceland_label=face_label
            rect_label=np.array(rect_label)
            faceland_label=np.array(faceland_label)
            rect_label[:,0]+=offsetx
            rect_label[:,1]+=offsety

            faceland_label[:,0]+=offsetx
            faceland_label[:,1]+=offsety

            global_face_label_list[i]=(list(rect_label),list(faceland_label))


    global_offset[0]=offsetx
    global_offset[1]=offsety


def global_reset():

    global_face_label_list.clear()
    global_facelandpts_label.clear()
    global_facerctpts_label.clear()
    for i,_ in enumerate(global_offset):
        global_offset[i]=0
    for i,_ in enumerate(global_extend_lenth_list):
        global_extend_lenth_list[i]=0
    for i,_ in enumerate(global_flags):
        global_flags[i]=0
        
    


def update_label_from_pred(plateland_net,imgfull,face_label_list_loaded):

    # face_label_list_loaded.append((list(facerect.astype(np.int32)),list(faceland.astype(np.int32))))
    face_label_list_loaded_new=[]
    for face_label in face_label_list_loaded:
        facelist,facelandlist=face_label
        # print('facelandlist:',facelandlist)
        # exit(0)

        facelandlist_new=pred_plateland(imgfull,np.array(facelandlist),plateland_net)

        plate_center=facelandlist_new.mean(axis=0).astype(np.int32)

        plateland5=[
            facelandlist_new[0],
            facelandlist_new[1],
            plate_center,
            facelandlist_new[2],
            facelandlist_new[3],
        ]

        face_label_list_loaded_new.append((facelist,list(plateland5)))
    return face_label_list_loaded_new


if __name__ == '__main__':
    # global global_face_label_list
    # global img


    plate_land_pt=r'/home/tao/disk1/Workspace/Project/Pytorch/TSLPR/plate_landmark/plateland.pt'
    plateland_net=torch.jit.load(plate_land_pt)

    # ims=get_ims(local_path)
    ims=get_dirims(local_path)
    ims.sort()


    easy_root=os.path.join(local_path,'refine')
    hard_root=os.path.join(local_path,'notuse')

    makedir(easy_root)
    makedir(hard_root)


    cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('img', mouse_func_facerect)

    for i,im in enumerate(ims):
        # if i==0:
        #     continue
        print(im)
        img=cv2.imread(im)
        global_img_info[0]=np.array(img)
        h,w,c=img.shape
        global_img_info[1]=h
        global_img_info[2]=w
        global_img_info[3]=c

        filekey,ext,path_noext=parse_path(im)
        txt_path=imgpath_to_txtpath(im)
        imname=os.path.basename(im)
        txtname=os.path.basename(txt_path)

        global_facelandpts_label.clear()
        global_facerctpts_label.clear()


        # img=cv2.imread(im)
        h,w,c=img.shape

        cv2.imshow('img',img)
        limit_window(img,'img')

        txt_path=imgpath_to_txtpath(im)
        HasLabel=False
        if os.path.exists(txt_path):
            HasLabel=True
            face_label_list_loaded=load_ano_from_txt(txt_path,w,h)
            face_label_list_loaded=update_label_from_pred(plateland_net,img,face_label_list_loaded)

            global_face_label_list=face_label_list_loaded.copy()
            update_extend_load()
            update_view_main()


        while 1:
            key=cv2.waitKey()
            print(key)
            if key==27:
                exit(0)
            if key in key_dic['ENTER']:
                ano_lines=make_facenao_lines()

                txt_path=imgpath_to_txtpath(im)
                # if HasLabel:
                #     print('keep label?',global_face_label_list==face_label_list_loaded)

                with open(txt_path,'w') as f:
                    f.writelines(ano_lines)
                print(ano_lines)

                shutil.move(im,os.path.join(easy_root,imname))
                if os.path.exists(txt_path):
                    shutil.move(txt_path,os.path.join(easy_root,txtname))

                global_reset()
                break

            if key ==ord('1'):
                shutil.move(im,os.path.join(hard_root,imname))
                if os.path.exists(txt_path):
                    shutil.move(txt_path,os.path.join(hard_root,txtname))

                global_reset()
                break

            if key ==ord('+') or key ==ord('2') or key ==ord('=') :
                cross_cursor_ind=global_flags[3]
                # print('cross_cursor_ind:',cross_cursor_ind)

                cind=(1+cross_cursor_ind)%4
                global_extend_lenth_list[cind]+=0.2
                update_extend()
                update_view_main()
                pass
            if key ==ord('3'):
                cross_cursor_ind=global_flags[3]
                # print('cross_cursor_ind:',cross_cursor_ind)
                cind=(1+cross_cursor_ind)%4
                if global_extend_lenth_list[cind]>=0.2:
                    global_extend_lenth_list[cind]-=0.2
                update_extend()
                update_view_main()
                pass



            if key in key_dic['BACK']:
                if len(global_facerctpts_label)>0:
                    global_facerctpts_label.pop()
                    # print('pop')
            if key in key_dic['DELETE']:
                if len(global_face_label_list)>0:
                    rect_cursor=global_flags[0]
                    del global_face_label_list[rect_cursor]
                    global_flags[0]=0
                print('delete')

            if key in key_dic['SPACE']:
                if len(global_face_label_list)>0:
                    rect_cursor=global_flags[0]
                    rect_label,faceland_label=global_face_label_list[rect_cursor]


                    tl,br=rect_label[0],rect_label[1]  
                    croprct=get_facecrop_rect(img,tl,br)
                    tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
                    faceland_label=np.array(faceland_label)
                    faceland_label[:,0]-=tlx
                    faceland_label[:,1]-=tly
                    # print(faceland_label)

                    global_facerctpts_label=rect_label
                    global_facelandpts_label=list(faceland_label)
                    del global_face_label_list[rect_cursor]
                    global_flags[0]=0
                    refine_face()

                print('space')

            update_view_main()

        global_face_label_list.clear()