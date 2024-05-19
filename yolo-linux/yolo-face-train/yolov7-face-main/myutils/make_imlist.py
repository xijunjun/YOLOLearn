import os,sys,cv2



def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

if __name__=='__main__':

    imroot=[
        # r'/home/tao/disk1/Dataset/WiderFace/ForYolo7Face/val/images'
        r'/home/tao/mynas/workspace/yoloface/dataset/WIDER_val/val/images'
    ][0]
    ims=get_ims(imroot)

    lines=''
    for im in ims:
        # imname=os.path.basename(im)
        # lines+=imname+'\n'

        relpath=os.path.relpath(im,imroot)

        # relpath=os.path.realpath(r'/mnt/mynas/workspace/yoloface/dataset/WIDER_val/val/images')
        lines+=relpath+'\n'



    lines=lines.rstrip('\n')

    txtpath=imroot+'.txt'
    with open(txtpath,'w') as f:
        f.writelines(lines)


    print('finish')