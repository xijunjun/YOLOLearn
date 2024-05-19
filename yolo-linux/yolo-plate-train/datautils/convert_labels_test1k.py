#coding:utf-8
import  numpy as np
import  cv2
import os,sys
import platform
import shutil


SCREEN_WIDTH=1800                    #图像窗口最大宽度
SCREEN_HEIGHT=900                    #图像窗口最大高度
local_path = u"/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/test1k/"
imgpathlist=os.listdir(local_path)

banlst=u'警港澳挂领使学字'
def limit_window(disimg,winnane):
    wm_ratio=1.0
    if disimg.shape[1] > SCREEN_WIDTH or disimg.shape[0] > SCREEN_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (SCREEN_WIDTH / float(SCREEN_HEIGHT)):
            cv2.resizeWindow(winnane, SCREEN_WIDTH, int(SCREEN_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = SCREEN_WIDTH / float(disimg.shape[1])
        else:
            cv2.resizeWindow(winnane, int(SCREEN_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), SCREEN_HEIGHT)
            wm_ratio = SCREEN_HEIGHT / float(disimg.shape[0])
    else:
        cv2.resizeWindow(winnane, disimg.shape[1], disimg.shape[0])
    return wm_ratio
def file_extension(path):
  return os.path.splitext(path)[1]
def _load_pascal_annotation(_data_path):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    lines=open(_data_path,encoding='gbk').readlines()
    num_objs=len(lines)
    boxes = np.zeros((num_objs, 8), dtype=np.uint16)
    platestr=[]
    is_usual = [1 for i in range(0,num_objs)]
    for i,oneline in enumerate(lines):
        # item= lines[i].decode('gbk').rstrip().split()
        item= lines[i].rstrip().split()
        # print type(item[0]),type(u'沪D71603')
        boxes[i, :] = [int(num) for num in item[0:8]]
        platestr.append((item[10]))
    return {'boxes': boxes,
            'platestr':platestr
            }
def encode_thr_sys(tstr):
    return tstr.encode('gbk') if 'Windows' in platform.system() else tstr.encode('utf-8')
def isval(oristr):
    for orichar in oristr:
        if orichar  in banlst:
            return False
    return True


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
    # print('x:',x)
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


def make_facenao_lines(oriimg,face_label_list):

    h,w,c=oriimg.shape
    lines=''
    for face_label in face_label_list:
        line='0 '
        rect_label,faceland_label=face_label

        rect_label_xywh=np.array(rect_label)
        faceland_label=np.array(faceland_label)

        # print('rect_label_xywh:',rect_label_xywh)
        rect_label_xywh=xyxy2xywh(rect_label_xywh)
        for pt in rect_label_xywh:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' '

        for pt in faceland_label:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' 2.0 '
        line=line.rstrip(' ')+'\n'
        lines+=line
    lines=lines.rstrip('\n')
    return lines


def get_imkey_ext(imname):
    imname=os.path.basename(imname)
    ext='.'+imname.split('.')[-1]
    imkey=imname.replace(ext,'')
    return imkey,ext

def pts2rct(box):
    tlx = min(box[:,0])
    tly = min(box[:,1])
    brx = max(box[:,0])
    bry = max(box[:,1])
    return [tlx,tly,brx,bry]

if __name__ == '__main__':

    dstroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataTest1k'

    txtroot=r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/labels_yolov7/test1k'

    prefix='test1k'


    numims=len(imgpathlist)
    for idx,i_img in enumerate(imgpathlist):
        print('{}of{}'.format(idx,numims))

        if  file_extension(i_img) not in ['.JPG','.bmp','.PNG','.png','.jpeg','.jpg']:
            continue

        face_label_list=[]
        impath=local_path+i_img
        txtpath=impath.split('.')[0]+'.txt'
        # picano=_load_pascal_annotation(impath.split('.')[0]+'.txt')
        print('txtpath:',txtpath)
        imname=os.path.basename(i_img)
        


        picano=_load_pascal_annotation(txtpath)
        # ori_img = cv2.imread(encode_thr_sys(local_path + i_img))
        ori_img = cv2.imread(local_path + i_img)
        imgpath=local_path + i_img

        boxes=picano['boxes']

        bdboxes=[]
        
        for i in range(boxes.shape[0]):
            quad=[]
            for j in range(4):
                cv2.line(ori_img, (boxes[i][j*2], boxes[i][j*2+1]), (boxes[i][(j+1)%4*2], boxes[i][(j+1)%4*2+1]), (0, 0, 255), thickness=2)
                quad.append([boxes[i][j*2], boxes[i][j*2+1]])
            quad=np.array(quad)
            bdbox=pts2rct(quad)
            # bdboxes.append(bdbox)


            plate_land=[]
            plate_center=quad.mean(axis=0).astype(np.int32)
            plate_land=[]
            for j in range(4):
                if j==2:
                    plate_land.append(plate_center)
                plate_land.append(quad[j])
            plate_land=np.array(plate_land)


            bdbox=np.array(bdbox)
            bdbox=bdbox.reshape((2,2))
            print('plate_land:',plate_land)

            face_label_list.append((bdbox,plate_land))

        for pstr in picano['platestr']:
            print (pstr,)
        print ('')


        anolines=make_facenao_lines(ori_img,face_label_list)

        newimname=prefix+'-'+str(idx).zfill(5)+'.jpg'
        newimpath=os.path.join(dstroot,newimname)
        # cv2.imwrite(newimpath,ori_img)
        shutil.copy(imgpath,newimpath)

        imkey,ext=get_imkey_ext(newimname)
        txtdstpath=os.path.join(txtroot,imkey+'.txt')
        with open(txtdstpath,'w') as f:
            f.writelines(anolines)



        # # print picano['platestr']
        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # limit_window( ori_img,'img')
        # cv2.imshow('img',ori_img)

        # key=cv2.waitKey()
        # if key==27:
        #     break