from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform
import math



MAX_WIDTH=1800                    #图像窗口最大宽度
MAX_HEIGHT=1000                    #图像窗口最大高度

def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

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


def makedir(filedir):
    if os.path.exists(filedir) is False:
        os.mkdir(filedir)

def xywh2xyxy(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
    return y

def load_ano_from_txt(txtpath,w,h):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    face_label_list_loaded=[]
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




        facerect[:,0]*=w
        facerect[:,1]*=h
        faceland[:,0]*=w
        faceland[:,1]*=h

        print(facerect)

        face_label_list_loaded.append((list(facerect.astype(np.int32)),list(faceland.astype(np.int32))))

        # break
    return face_label_list_loaded

if __name__=='__main__':

    # imroot=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\refine'
    imroot=r'Z:\workspace\yoloface\dataset\MAFA\MAFA1000\yuejie\refine\2'
    ims=get_ims(imroot)
    ims.sort()

    # easy_root=os.path.join(imroot,'easy')
    # hard_root=os.path.join(imroot,'hard')

    # makedir(easy_root)
    # makedir(hard_root)

    global_face_label_list=None


    cv2.namedWindow('img', cv2.WINDOW_FREERATIO)

    for im in ims:
        img=cv2.imread(im)
        filekey,ext,path_noext=parse_path(im)
        txt_path=imgpath_to_txtpath(im)
        imname=os.path.basename(im)
        txtname=os.path.basename(txt_path)


        h,w,c=img.shape

        
        if os.path.exists(txt_path):
            face_label_list_loaded=load_ano_from_txt(txt_path,w,h)
            global_face_label_list=face_label_list_loaded.copy()


        rectline_thick_base=2.0
        circle_r_base=3.0
        circle_r_offset=2.0
        # rectline_thick_base

        disimg=np.array(img)
        linethick_scale_ratio=cac_winratio(disimg,MAX_WIDTH,MAX_HEIGHT)
        # print('linethick_scale_ratio:',linethick_scale_ratio)
        line_ratio_mul=1/linethick_scale_ratio


        if len(global_face_label_list)>0:
            print('len(global_face_label_list):',len(global_face_label_list))
            for face_label in global_face_label_list:
                rect_label,faceland_label=face_label
                for pt in faceland_label:
                    cv2.circle(disimg, (pt[0], pt[1]),int(circle_r_offset+circle_r_base*line_ratio_mul+0.5), (0, 0, 255), thickness=-1)

                tl=rect_label[0]
                br=rect_label[1]   
                cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), int(rectline_thick_base*line_ratio_mul+0.5))

        cv2.imshow('img',disimg)
        limit_window(disimg,'img')

        key=cv2.waitKey(0)
        # if key==13:#enter
        #     shutil.move(im,os.path.join(easy_root,imname))
        #     if os.path.exists(txt_path):
        #         shutil.move(txt_path,os.path.join(easy_root,txtname))
        # if key==32:
        #     shutil.move(im,os.path.join(hard_root,imname))
        #     if os.path.exists(txt_path):
        #         shutil.move(txt_path,os.path.join(hard_root,txtname)) 

        if key==27:
            exit(0)



    print('finish')