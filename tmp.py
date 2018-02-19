#-*- coding:utf-8 -*-
import glob
import os
import shutil
import numpy as np
csv_files=glob.glob('./data/fundus_rois/*.csv')
data_dir = './data/fundus_roidb'
img_dir = './data/fundus_images'
img1_dir= './data/fundus_images_1'
#npy 파일 만들기.
#fundus roidb에는 있는데 fundus_images는 없는 이미지들이 있다 .
# img을 먼저 불러오고 그 다음 fundus roidb을 찾아서 오류가 난다
# 그래서 roi가 있는 이미지만 fundus_imags1 에 옮겼다
a=[0.7,0.3,0.9]
a=np.asarray(a)
b=a.argsort()[::-1]
print b[0]
print a[b]

'./text.txt'
print os.path.splitext('./text.txt')
print os.path.split('./abc/text.txt')
exit()


for csv_file in csv_files:
    name = os.path.splitext(os.path.split(csv_file)[-1])[0]
    f=open(csv_file , 'r')
    coords=[]
    for line in f.readlines()[1:]:
        x1,y1,x2,y2=line.split(',')[1:]
        y2=y2.replace('\n','')
        x1,y1,x2,y2=map(float , [x1,y1,x2,y2])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        coords.append([x1,y1,x2,y2])
    coords=np.asarray(coords)
    save_path = os.path.join(data_dir,os.path.splitext(os.path.split(csv_file)[-1])[0] + '.png')
    np.save(save_path, coords)
    shutil.copy(os.path.join(img_dir , name+'.png') , os.path.join(img1_dir , name+'.png'))
    print np.load(save_path+'.npy')









