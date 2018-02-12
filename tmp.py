import glob
import os
import numpy as np
csv_files=glob.glob('./data/fundus_rois/*.csv')
data_dir = './data/fundus_roidb'

for csv_file in csv_files:
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
    np.save(save_path, csv_file)







