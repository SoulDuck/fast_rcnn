from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np
import os
import xml.etree.ElementTree as ET
import urllib.request, urllib.parse, urllib.error
import tarfile
import shutil
from glob import glob

PASCAL_VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

def parse_pascal_labels(annotation_filepath):
    res = []
    classes = []
    difficulities = []
    tree = ET.parse(annotation_filepath)
    for el in tree.findall('object'):
        cls = el.find('name').text
        box = el.find('bndbox')
        x1 = float(box.find('xmin').text)
        y1 = float(box.find('ymin').text)
        x2 = float(box.find('xmax').text)
        y2 = float(box.find('ymax').text)
        diff = el.find('truncated').text
        res.append([x1, y1, x2, y2])
        classes.append(cls)
        difficulities.append(int(diff))
    return np.asarray(res), np.asarray(classes), np.asarray(difficulities)

def extract_images(pascal_dir, output_dir):
    img_dir = os.path.join(output_dir, 'images')
    roidb_dir = os.path.join(output_dir, 'roidb')
    if os.path.isdir(img_dir) or os.path.isdir(roidb_dir):
        print('output directories already exist, stopping')
        return
    os.makedirs(img_dir)
    os.makedirs(roidb_dir)
    ann_paths = sorted(glob(os.path.join(pascal_dir, 'Annotations')+"/*"))
    im_paths = sorted(glob(os.path.join(pascal_dir, 'JPEGImages')+"/*"))
    mu = 0
    for (ann_path, im_path) in zip(ann_paths, im_paths):
        boxes, classes, diff = parse_pascal_labels(ann_path)
        if 'car' in classes:
            logi = np.logical_and(classes=='car', diff==0)
            boxes = boxes[logi]
            if len(boxes) == 0:
               continue
            shutil.copy(im_path, os.path.join(img_dir, str(mu) + '.jpg'))
            np.save(os.path.join(roidb_dir, str(mu) + '.jpg.npy'), boxes)
            mu += 1

def main():
    print("Downloading VOC dataset")
    urllib.request.urlretrieve(PASCAL_VOC_URL, "VOCtest.tar")
    print("Extracting VOC dataset")
    with tarfile.open("VOCtest.tar") as tar:
        tar.extractall("../VOCdata")
    print("Selecting car images")
    extract_images("../VOCdata/VOCdevkit/VOC2007", "../data")
    print("Removing downloaded data")
    os.remove("VOCtest.tar")
    shutil.rmtree("../VOCdata")

if __name__ == '__main__':
    main()
