from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from trainer import Trainer
def main():
    net_trainer = Trainer('None')
    net_trainer.connect_roi_pooling()
    net_trainer.connect_detector()
    net_trainer.fit(200 , 'deleteme')

if __name__ == '__main__':
    main()
