from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from trainer import Trainer
def main():
    net_trainer = Trainer()  # initializes imagenet pretrained convnet
    net_trainer.connect_roi_pooling()  # connects pretrained layer with roi_pooling layer
    net_trainer.connect_detector()  # connects detector to roi_pooling_layer
    net_trainer.fit(200)  # runs training of the network - you can observe the progress in Neptune

if __name__ == '__main__':
    main()
