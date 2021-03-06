#-*- coding:utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import cv2
from fast_rcnn import FastRCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import eval

class Trainer(object):
    def __init__(self , data_type):
        if data_type =='fundus':
            self.im_dir = './data/fundus_images'
            self.roi_dir = './data/fundus_roidb'
            self.img_h = 1001
            self.img_w = 1001

        else:
            self.im_dir = './data/images'
            self.roi_dir = './data/roidb'
            self.img_h=1920
            self.img_w=576

        im_paths = self.get_im_paths()
        roi_paths = self.get_roi_paths()
        self.show_im =True
        self.pretrained_path = './data/vgg16-20160129.tfmodel'
        self.im_paths = np.asarray(im_paths)
        self.roi_paths = np.asarray(roi_paths)
        self.net = FastRCNN(path=self.pretrained_path)


    def get_im_paths(self):
        self.im_names = os.listdir(self.im_dir)
        paths = [os.path.join(self.im_dir, el) for el in self.im_names]
        return paths

    def get_roi_paths(self):
        paths = [os.path.join(self.roi_dir, el + '.npy') for el in self.im_names]
        return paths

    def generate_positive_roi(self, gt_boxes):
        res = []
        num_ = []
        for j in range(32):
            epsilon1 = np.random.randint(-10, 10)
            epsilon2 = np.random.randint(-10, 10)
            num_box = np.random.randint(0, len(gt_boxes))
            bxes = gt_boxes[num_box] + np.asarray([epsilon1, epsilon2, epsilon1, epsilon2])
            bxes /= 16 # this box will apply for feature map
            res.append(bxes)
            num_.append(num_box)
        return np.asarray(res), np.asarray(num_)

    def get_iou(self, pred_bbox , gt_bboxes):
        #여러개의 gt박스가 있으면 가장 많이 겹치는 gt_bbox이다

        ious=[]
        for gt_bbox in gt_bboxes:
            p_x1, p_y1, p_x2, p_y2 = pred_bbox
            g_x1, g_y1, g_x2, g_y2 = gt_bbox

            xx1 = np.maximum(p_x1, g_x1)
            yy1 = np.maximum(p_y1, g_y1)
            xx2 = np.minimum(p_x2, g_x2)
            yy2 = np.minimum(p_y2, g_y2)

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap_area = w*h
            pred_area=(p_x2-p_x1+1)*(p_y2-p_y1+1)
            gt_area=(g_x2-g_x1+1)*(g_y2-g_y1+1)

            iou=overlap_area/float(pred_area + gt_area - overlap_area)
            ious.append(iou)
        return np.max(ious)

    """
        def generate_negative_roi(self, gt_boxes ):
            res = []
            for j in range(96):
                epsilon1 = np.random.randint(50, 300)
                epsilon2 = np.random.randint(50, 300)
                num_box = np.random.randint(0, len(gt_boxes))
                bxes = gt_boxes[num_box] + np.asarray([epsilon1, epsilon2, epsilon1 + 100, epsilon2 + 75])
                bxes /= 16# it'll apply for feature map
                res.append(bxes)
            return np.asarray(res)
        """

    def generate_negative_roi(self, gt_boxes , im_dims):
        res = []
        h,w=im_dims
        n_neg = 96
        count=0
        while True:
            xx = random.sample(range(h), 2)
            x2 = np.max(xx)
            x1 = np.min(xx)

            yy = random.sample(range(w), 2)
            y2 = np.max(yy)
            y1 = np.min(yy)
            if self.get_iou([x1,y1,x2,y2] , gt_bboxes=gt_boxes) > 0:
                continue;

            bxes = np.asarray([x1,y1,x2,y2])
            bxes /= 16# it'll apply for feature map
            res.append(bxes)
            if len(res) == n_neg:
                break;
            count+=1
        return np.asarray(res)


    def relabel(self, gt_boxes, old_shape, new_shape):
        mult = np.asarray([float(new_shape[1])/old_shape[1],
                           float(new_shape[0])/old_shape[0],
                           float(new_shape[1])/old_shape[1],
                           float(new_shape[0])/old_shape[0]])
        bxes = np.copy(gt_boxes)
        bxes = np.float32(bxes)
        for el in bxes:
            el *= mult
        return bxes

    def bbox_transform(self, boxes, shape):
        boxes_ = np.copy(boxes)
        boxes_ = np.float32(boxes_)
        dw = 1./shape[1]
        dh = 1./shape[0]

        x = (boxes_[:, 0] + boxes_[:, 2])*dw/2.
        y = (boxes_[:, 1] + boxes_[:, 3])*dh/2.
        w = (boxes_[:, 2] - boxes_[:, 0])*dw
        h = (boxes_[:, 3] - boxes_[:, 1])*dh
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        w = np.reshape(w, (-1, 1))
        h = np.reshape(h, (-1, 1))
        return np.hstack([x, y, w, h])

    def transform_inv(self, output, shape):
        dx = shape[1]
        dy = shape[0]
        x = output[:, 0]*dx
        y = output[:, 1]*dy
        w = output[:, 2]*dx
        h = output[:, 3]*dy
        x -= w/2.
        y -= h/2.
        x2 = x + w
        y2 = y + h
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        x2 = np.reshape(x2, (-1, 1))
        y2 = np.reshape(y2, (-1, 1))
        return np.hstack([x, y, x2, y2])

    def connect_roi_pooling(self):
        self.net.add_roi_pooling()

    def connect_detector(self):
        self.net.connect_detector()

    def get_top_k(self, boxes, scores, k):
        a = np.argsort(np.squeeze(scores))[::-1]
        return boxes[a[:k]]

    # by Ross Girshick
    def non_maximum_supression(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]]) # xx1 shape : [19,]
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h #inter shape : [ 19,]
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def send_image_with_proposals(self, im, proposals, shape, rois=False):
        width = 300
        height = 300
        im_ = cv2.resize(im, (width, height))
        im_ = np.uint8(im_ * 255.)
        for proposal in proposals:
            x1 = int(width * proposal[0] / float(shape[1]))
            y1 = int(height * proposal[1] / float(shape[0]))
            x2 = int(width * proposal[2] / float(shape[1]))
            y2 = int(height * proposal[3] / float(shape[0]))
            cv2.rectangle(im_, (x1, y1), (x2, y2), (255, 0, 0), 1)
        pil_im = Image.fromarray(im_)
        count = 0

        if not rois is None:
            for roi in rois:
                x1 = int(width * roi[0] / float(shape[1]))
                y1 = int(height * roi[1] / float(shape[0]))
                x2 = int(width * roi[2] / float(shape[1]))
                y2 = int(height * roi[3] / float(shape[0]))
                cv2.rectangle(im_, (x1, y1), (x2, y2), (0, 0, 255), 1)
            pil_im = Image.fromarray(im_)
            plt.imshow(pil_im)
            while (True):
                save_path = os.path.join('./tested_images', str(count) + '.png')
                if not os.path.isfile(save_path):
                    plt.imsave(fname = save_path , arr= pil_im)
                    print 'images saved!'
                    break;
                else:
                    count += 1

            plt.show()
            plt.close()
            # neptune_im = neptune.Image(name='all the RoIs', description='region proposals', data=pil_im)
            # self.im_channels[0].send(x=time_step, y=neptune_im)

        else:
            plt.imshow(pil_im)
            while(True):
                save_path = os.path.join('./tested_images', str(count) + '.png')
                if not os.path.isfile(save_path):
                    plt.imsave(fname=save_path, arr=pil_im)
                    print 'images saved!'
                    break;
                else:
                    count+=1
            plt.show()
            plt.close()
            pass;
            # neptune_im = neptune.Image(name='chosen RoIs', description='object detections', data=pil_im)
            #self.im_channels[1].send(x=time_step, y=neptune_im)

    def fit(self , num_epochs , save_path):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
            # Session start
            saver=tf.train.Saver(max_to_keep=10)
            save_dir , model_name =os.path.split(save_path)
            init = tf.global_variables_initializer()
            sess.run(init)
            # Restore Model
            if tf.train.get_checkpoint_state(checkpoint_dir=save_dir):
                save_path=os.path.join(save_path)
                saver.restore(sess, save_path=save_path)
            k = 0
            losses_ = [[], [], []]
            global_step=0

            for i in range(num_epochs):
                ids = np.arange(len(self.im_paths))
                np.random.shuffle(ids)
                im_paths = self.im_paths[ids]
                roi_paths = self.roi_paths[ids]

                for m in range(len(im_paths)):
                    # Load image and Resize image
                    im = cv2.imread(im_paths[m]).astype('float32')/255.
                    gt_boxes = np.load(roi_paths[m])
                    gt_boxes = self.relabel(gt_boxes, im.shape, (self.img_w, self.img_h))
                    im = cv2.resize(im, (self.img_h, self.img_w))
                    x = im.reshape((1, im.shape[0], im.shape[1], 3))

                    # Get positive and Negative roi
                    positive_rois, ob_numbers = self.generate_positive_roi(gt_boxes)
                    negative_rois = self.generate_negative_roi(gt_boxes,(self.img_h , self.img_w))

                    #self.show_imgae_with_rois(im , positive_rois , negative_rois , gt_boxes)

                    # change ROI shape [32+96, 4] --> [32+96, 5]
                    rois = np.vstack([positive_rois, negative_rois])
                    y_ = np.zeros((len(rois), ), dtype=np.int32)

                    zeros = np.zeros((len(rois), 1))
                    rois = np.hstack([zeros, rois])
                    rois = np.int32(rois)
                    y_[:len(positive_rois)] = 1
                    reg = np.zeros((len(rois), 4))
                    boxes_tr = self.bbox_transform(gt_boxes, im.shape)

                    for j in range(32):
                        reg[j] = boxes_tr[ob_numbers[j]]
                        print boxes_tr[ob_numbers[j]]

                    feed_dict = {self.net.x_: x, self.net.y_bbox: reg, self.net.y_cls: y_, self.net.roidb: rois,
                                 self.net.lr: 0.0001,
                                 self.net.im_dims: [[self.img_h, self.img_w]]}

                    _ , boxes =sess.run([self.net.opt ,self.net.boxes] , feed_dict=feed_dict)
                    tot, reg, cls = sess.run([self.net.loss_, self.net.reg_loss, self.net.class_loss],
                                             feed_dict=feed_dict)



                    losses_[0].append(reg)
                    losses_[1].append(cls)
                    losses_[2].append(tot)

                    if m % 100 == 0:
                        reg = sum(losses_[0])/len(losses_[0])
                        cls = sum(losses_[1])/len(losses_[1])
                        tot = sum(losses_[2])/len(losses_[2])
                        print reg , cls ,tot

                    if m % 1 == 0:
                        try:
                            boxes , logits = sess.run([self.net.boxes ,self.net.logits], feed_dict=feed_dict)
                            boxes = self.transform_inv(boxes, im.shape)

                            # Get Scores
                            logits = logits[:, 1]
                            scores = np.reshape(logits, (-1, 1))

                            boxes = np.hstack([boxes, scores])
                            boxes = self.get_top_k(boxes, logits, 50)
                            keep = self.non_maximum_supression(boxes, .3)
                            boxes = boxes[keep, :4]

                            # Roi Pooling
                            roi_pool = rois[:, 1:]*np.asarray([16, 16, 16, 16])
                            self.send_image_with_proposals(k, im[:, :, [2, 1, 0]], boxes, im.shape , gt_boxes)
                            saver.save(sess , save_path='./saved_model/model' , global_step=global_step)
                            #self.handler.send_image_with_proposals(k, im[:, :, [2, 1, 0]], roi_pool, im.shape, True)
                            k += 1
                        except Exception as e :
                            print e
                            pass;

                    global_step+=1

    def show_imgae_with_rois(self , im , pos_rois , neg_rois , gt_bboxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for r in pos_rois:
            x1, y1, x2, y2 = r * 16
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b')
            ax.add_patch(rect)

        for box in neg_rois:
            x1, y1, x2, y2 = box * 16
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r')
            ax.add_patch(rect)

        for box in gt_bboxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='w')
            ax.add_patch(rect)

        plt.imshow(im)
        plt.show()
if '__main__' == __name__:
    import matplotlib.patches as patches
    sample_gtboxes=np.load('/Users/seongjungkim/PycharmProjects/fast_rcnn/data/fundus_roidb/5068136_20160511_R.png.npy')
    img=Image.open('/Users/seongjungkim/PycharmProjects/fast_rcnn/data/fundus_images/5068136_20160511_R.png')
    trainer=Trainer('fundus')
    pos_res , pos_num =trainer.generate_positive_roi(sample_gtboxes)
    neg_res = trainer.generate_negative_roi(sample_gtboxes,(1001,1001))
    trainer.show_imgae_with_rois(img ,pos_res , neg_res , sample_gtboxes)
