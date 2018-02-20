# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os


class Eval():
    def __init__(self ):
        print 'eval instance'
    def restore_model(self , model_path):
        # restore model
        self.sess = tf.Session(config=tf.ConfigProto, allow_soft_placement=True, log_device_placement=False)
        saver = tf.train.import_meta_graph(meta_graph_or_file=os.path.splitext(model_path)[0] + '.meta')
        saver.restore(self.sess, save_path=os.path.splitext(model_path)[0])
        self.boxes = tf.get_default_graph().get_tensor_by_name('boxes:0')
        self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')
        self.x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
        self.y_cls = tf.get_default_graph().get_tensor_by_name('y_cls:0')
        self.y_bboxes = tf.get_default_graph().get_tensor_by_name('y_bboxes:0')

    def get_scores(self ,sess ,logits_tensor,image):
        scores=sess.run(logits_tensor, feed_dict={self.x_ : image})
        return scores[:,1]

    def get_boxes(self,sess,boxes_tensor,image):
        bboxes = sess.run(self.boxes, boxes_tensor,feed_dict={self.x_: image})

        # when corresponding original image with bboxes , you must do 'transform_inv (bboxes , im_dims)'
        return bboxes

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

    def get_groundtruths(self, ious , treshold):
        return [np.asarray(ious) > treshold]

    def get_recall_precision(self,scores , groundtruths):
        assert len(scores) == len(groundtruths)
        scores_gts = np.hstack((np.asarray(scores).reshape((-1, 1)), np.asarray(groundtruths).reshape((-1, 1))))  # scr = scores
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
            precision = n_true_pos / (i+1)

            ret_recall.append(recall)
            ret_precision.append(precision)
        return ret_recall , ret_precision


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

    def get_interpolated_precision(self,recall , precision , thres_recall):
        #print recall_precision
        #recall = recall_precision[:, 0]
        #precision = recall_precision[:, 1]
        recall , precision =map(np.asarray , [recall , precision])
        indices=[recall >= thres_recall]
        indices=np.where(indices)[1]
        selected_positive=precision[indices]
        if len(selected_positive)==0:
            return 0
        return np.max(selected_positive)

    def get_AP(self,recall,precision): # AP = Average Precision
        precisions=[]
        for i in range(11):
            thres_recall=i/10.
            precisions.append(self.get_interpolated_precision(recall, precision, thres_recall))
        return np.mean(precisions)

    def get_mAP(self,images , gt_boxes , sess, logits_tensor , boxes_tensor):
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
            scores=self.get_scores(sess,  logits_tensor ,img )
            pred_bboxes=self.get_bboxes(sess, boxes_tensor , img)
            ious=[]
            for bbox in pred_bboxes:
                iou=self.get_iou(bbox, gt_boxes)
                ious.append(iou)
            h,w=np.shape(img)[:2]
            pred_bbox=self.transform_inv(pred_bbox, (h, w))
            gt = self.get_groundtruths(ious=ious , treshold=0.5)
            recall , precision = self.get_recall_precision(scores, groundtruths=gt)
            ap = self.get_AP(recall , precision)
            mAP.append(ap)

        return np.mean(mAP)

if __name__ =='__main__':
    eval=Eval()
    ious = [0.78, 0.88, 0.76, 0.43, 0.44]
    pred_bbox = [0,0,10,10]
    gt_bboxes=[[5,5,15,15],[7,7,15,15]]
    print eval.get_iou(pred_bbox , gt_bboxes)

    scores=[0.97,0.96,0.94,0.93,0.91,0.89,0.78,0.77,0.76,0.56]
    gt= [True,True,True,False,True,True,False,True,True,True]
    recall,precision =eval.get_recall_precision(scores , gt)
    ap=eval.get_AP(recall , precision)
    print ap
