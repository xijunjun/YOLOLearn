
#coding:utf-8
import  numpy as np
import  cv2,random
import os,sys,shutil
import platform
# from  gen_plate_std  import plate_temp_list,get_stand_plate,char_dict
SCREEN_WIDTH=1800                    #图像窗口最大宽度
SCREEN_HEIGHT=900                    #图像窗口最大高度
# local_path = u"/disks/disk0/Dataset/Project/LPR/origindata/orisizeData/platedata30k/云/"
# imgpathlist=os.listdir(local_path)


mapdic={}
with open('index2char.txt') as mapf:
    for line in mapf.readlines():
        item=line.rstrip('\n').split(' ')
        mapdic[item[1]]=item[0]

def is_str_valid(platestr):
    if len(platestr)!=7:
        return False
    for pstr in platestr:
        if pstr not in mapdic.keys():
            return False
    return True


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

def limit_img_auto(img):
    sw=1920*1.0
    sh=1080*1.0
    h,w,c=img.shape
    swhratio=1.0*sw/sh
    whratio=1.0*w/h
    if whratio>swhratio:
        th=int(sh)
        if th>h:
            return img
        tw=int(w*(th/h))
        img=cv2.resize(img,(tw,th))
    else:
        tw=int(sw)
        if tw>w:
            return img
        th=int(h*(tw/w))
        img=cv2.resize(img,(tw,th))
    return  img


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def load_unify_label(txtpath,uselines=False):
    with open(txtpath,'r') as f:
        lines=f.readlines()
    imnames=[]
    bdrctslist=[]
    quadslist=[]
    has_quadlist=[]
    platetypeslist=[]
    platestrslist=[]

    for line in lines:
        line=line.rstrip('\n')
        items=line.split(';')
        imnames.append(items[0])

        platestrs=[]
        platetypes=[]

        platenum=int(items[3])
        bdrcts=[]
        quads=[]
        has_quad = False

        for i in range(0,platenum):
            platestr=items[4+i]
            plateitems=platestr.split(',')

            bdrctitems=plateitems[0].split(' ')
            bdrct=[]
            for tpstr in bdrctitems:
                bdrct.append(int(tpstr))
            bdrcts.append(bdrct)

            quad = []
            if plateitems[1]!='NULL':
                quaditems=plateitems[1].split(' ')
                for tpstr in quaditems:
                    quad.append(int(tpstr))
                has_quad=True
            quads.append(quad)

            platetypes.append(int(plateitems[3]))
            platestrs.append(plateitems[2])


        platetypeslist.append(platetypes)
        platestrslist.append(platestrs)

        bdrctslist.append(bdrcts)
        quadslist.append(quads)
        has_quadlist.append(has_quad)

    return {'imnames':imnames,
            'bdrctslist':bdrctslist,
            'lines':lines,
            'has_quadlist':has_quadlist,
            'quadslist':quadslist,
            'platetypeslist':platetypeslist,
            'platestrslist':platestrslist
            }

def getpath(rootlist,imname):
    for root in rootlist:
        if os.path.exists(os.path.join(root,imname)):
            return os.path.join(root,imname)
    return None


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


def make_facenao_lines(oriimg,face_label_list):

    h,w,c=oriimg.shape
    lines=''
    for face_label in face_label_list:
        line='0 '
        rect_label,faceland_label=face_label

        rect_label_xywh=np.array(rect_label)
        faceland_label=np.array(faceland_label)


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

if __name__ == '__main__':
    txtpathlist=[
                # '/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00_unifylabel.txt',
                # '/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch01_unifylabel.txt'
                r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00_unifylabel.txt'
                 ]
    rootlist=[
            '/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00',
            # '/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch01'
    ]
    dstroot=[
        # r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/labels/labels_yolov7'
        # r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/platedata_yolo_train/AnyPlateDataBatch00/labels'
        r'/home/tao/disk1/Dataset/Project/LPR/origindata/orisizeData/AnyplateData/AnyPlateDataBatch00'
    ][0]


    unify_label_dict={}
    for txtpath in txtpathlist:
        unify_label_dict.update(load_unify_label(txtpath))






    imnames=unify_label_dict['imnames']
    bdrctslist=unify_label_dict['bdrctslist']
    lines = unify_label_dict['lines']
    has_quadlist=unify_label_dict['has_quadlist']
    quadslist=unify_label_dict['quadslist']
    platetypeslist=unify_label_dict['platetypeslist']
    platestrslist=unify_label_dict['platestrslist']

    indlist=[i for i in range(0,len(imnames))]
    # random.shuffle(indlist)

    numims=len(imnames)
    for i in indlist:
        print('{}of{}'.format(i,numims))

        # print(i)
        imname=imnames[i]

        imkey,ext=get_imkey_ext(imname)

        impath=getpath(rootlist,imname)
        img=cv2.imread(impath)
        # print(lines[i])

        face_label_list=[]
        for k,bdrct in enumerate(bdrctslist[i]):
            plate_rct=np.array([[bdrct[0],bdrct[1]],[bdrct[2],bdrct[3]]])
            plate_land=[]

            cv2.rectangle(img, (bdrct[0],bdrct[1]),(bdrct[2],bdrct[3]), (255, 0, 0), 4)
            # print(platetypeslist[i][k], '--', platestrslist[i][k])

            # 关键点
            if has_quadlist[i]==True:
                quad=quadslist[i][k]
                # print('quad:',quad)
                for j in range(4):
                    cv2.line(img, (quad[j*2], quad[j*2+1]), (quad[(j+1)%4*2], quad[(j+1)%4*2+1]), (0, 0, 255), thickness=2)
                    plate_land.append([quad[j*2], quad[j*2+1]])
                
                plate_land=np.array(plate_land)
                plate_center=plate_land.mean(axis=0).astype(np.int32)
                # print('plate_center:',plate_center)
                cv2.circle(img, tuple(list(plate_center)), 2,(0, 255, 255), thickness=2)
                # cv2.circle(img, (100,100), 50, (255,255,0), 4)

                
                plate_land=[]
                for j in range(4):
                    if j==2:
                        plate_land.append(plate_center)
                    plate_land.append([quad[j*2], quad[j*2+1]])
                plate_land=np.array(plate_land)

                # print('plate_land before:',plate_land)
                # np.insert(plate_land,2,plate_center,axis=0)
                # print('plate_land after:',plate_land)
                for j in range(3):
                    cv2.line(img, (plate_land[j][0], plate_land[j][1]), (plate_land[(j+1)%5][0], plate_land[(j+1)%5][1]), (255, 255, 255), thickness=4)


                face_label_list.append((plate_rct,plate_land))
        anolines=make_facenao_lines(img,face_label_list)
        # print('anolines:',anolines)


        # print('-----')

        # if has_quadlist[i]==True:
        #     for quad in quadslist[i]:
        #         # print(quad)
        #         for j in range(4):
        #             cv2.line(img, (quad[j*2], quad[j*2+1]), (quad[(j+1)%4*2], quad[(j+1)%4*2+1]), (0, 0, 255), thickness=2)

        yololabel_path=os.path.join(dstroot,imkey+'.txt')
        # print('txtpath:',txtpath)


        # cv2.imshow('img',limit_img_auto(img))
        # key=cv2.waitKey(0)
        # if key==27:
        #     exit(0)
        with open(yololabel_path,'w') as f:
            f.writelines(anolines)



    print('finish')