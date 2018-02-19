# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os

class eval():
    def __init__(self ,model_path):
        #restore model
        self.sess=tf.Session(config=tf.ConfigProto,allow_soft_placement=True,log_device_placement=False)
        saver = tf.train.import_meta_graph(meta_graph_or_file=os.path.splitext(model_path)[0] +'.meta')
        saver.restore(self.sess , save_path=os.path.splitext(model_path)[0])
        self.boxes = tf.get_default_graph().get_tensor_by_name('boxes:0')
        self.logits= tf.get_default_graph().get_tensor_by_name('logits:0')
        self.x_= tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_cls= tf.get_default_graph().get_tensor_by_name('y_cls:0')
        self.y_bboxes = tf.get_default_graph().get_tensor_by_name('y_bboxes:0')

    def get_scores(self ,image):
        scores=self.sess.run(self.logits , feed_dict={self.x_ : image})
        return scores
    def get_boxes(self,image):
        bboxes = self.sess.run(self.boxes, feed_dict={self.x_: image})
        return bboxes

    def get_iou(self, pred_bbox , gt_bbox):
        #여러개의 gt박스가 있으면 가장 많이 겹치는 gt_bbox이다

        p_x1, p_y1, p_x2, p_y2 = pred_bbox
        g_x1, g_y1, g_x2, g_y2 = gt_bbox

        xx1 = np.maximun(p_x1, g_x1)
        yy1 = np.maximun(p_y1, g_y1)
        xx2 = np.minimun(p_x2, g_x2)
        yy2 = np.minimum(p_y2, g_y2)

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap_area = w*h

        pred_area=(p_x2-p_x1+1)*(p_y2-p_y1+1)
        gt_area=(g_x2-g_x1+1)*(g_y2-g_y1+1)
        iou=overlap_area/(pred_area + gt_area - overlap_area)
        return iou





    def get_groundtruths(self, ious , treshold):
        return ious > treshold

    #scores = get_scores(image)
    #pred_bboxes = get_bboxes(image)

    """
    ious=[]
        for i in range(len(pred_bboxes)):
            iou=get_iou(pred_bboxes , gt_boxes)
            ious.append(iou)
        gt=get_groundtruths(ious,  treshold=0.5)
    
    """
    def get_recall_precision(self,scores , groundtruths):
        assert len(scores) == len(groundtruths)
        scores_gts = np.hstack((scores.reshape((-1, 1)), groundtruths.reshape((-1, 1))))  # scr = scores
        order = scores_gts[:, 0].argsort()[::-1]
        scores_gts=scores_gts[order] # sort the list by ascenging order
        n_gts = len(scores_gts[: ,1] == True)

        n_true_pos=0.0
        ret_recall =[]
        ret_precision =[]
        for i , (scr , gt) in enumerate(scores_gts):
            if gt == True:
                n_true_pos +=1
            recall = n_true_pos / n_gts
            precision = n_true_pos / i+1
        ret_recall.append(recall)
        precision.append(precision)
        return ret_recall , ret_precision


    def get_interpolated_precision(self,recall_precision , thres_recall):
        recall = recall_precision[:, 0]
        precision = recall_precision[:, 1]
        indices=[recall >= thres_recall]
        selected_positive=precision[indices]
        return np.max(selected_positive)

    def get_AP(self,recall_precision): # AP = Average Precision
        precisions=[]
        for i in range(11):
            thres_recall=i/10.
            precisions.append(self.get_interpolated_precision(recall_precision, thres_recall))
        return np.mena(precisions)

    def get_mAP(self,images , gt_boxes):

        """
        example
        a=[0.7,0.3,0.9]
        a=np.asarray(a)
        b=a.argsort()[::-1]
        #>>>b = [2 0 1]
        print a[b]
        #>>>b = [ 0.9  0.7  0.3]
        """

        mAP=[]
        for img in images:
            scores=self.get_scores(img)
            pred_bboxes=self.get_bboxes(img)
            ious=self.get_ious(pred_bboxes , gt_boxes)
            gt = self.get_groundtruths(ious=ious , treshold=0.5)
            recall_precision = self.get_recall_precision(scores, groundtruths=gt)
            ap = self.get_AP(recall_precision)
            mAP.append(ap)

        return np.mean(mAP)