# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:23:08 2021

@author: William
"""










from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import re
import os
import glob
import os.path as osp

from torchreid.data import ImageDataset



class Wyze2(ImageDataset):
    dataset_dir = 'wyze2'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        #   in query/gallery).
        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(Wyze2, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path):
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        for img_path in img_paths:
            path_list = img_path.split(os.sep)
            img_name = path_list[6]
            pid, camid, img_num = img_name.split("_")
            
            pid_num = int(re.findall(r'\d+', pid)[0])
            camid_num = int(re.findall(r'\d+', camid)[0])
            
            data.append((img_path, pid_num, camid_num))
            
        return data











import torchreid





def main ():
    #register once and then comment out
    torchreid.data.register_image_dataset('wyze2', Wyze2)
    #Load data manager
    
    #transforms=['random_flip', 'random_crop']  removed transofrms to solve an error
    #datamanager = torchreid.data.ImageDataManager(
     #   root='reid-data',
      #  sources= 'ilidsvid',
       # targets= 'ilidsvid', #can make this "wyze2"
        #height=256,
        #width=128,
        #batch_size_train=32, #batch size 2 for my dataset
        #batch_size_test=100 #batch size 2 for my dataset
    #)
    
    datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='ilidsvid',
            height=256,
            width=128,
            batch_size_train=3,
            batch_size_test=3,
            seq_len=15,
            sample_method='evenly'
        )
    
    #Build model
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=datamanager.num_train_pids, #originally was this but to load a checkpoint we need to match the trained model class count datamanager.num_train_pids,
        loss='softmax',
        pretrained=False
    )
    
    #initialize model
    model = model.cuda()
    
    
    #setup optimizer
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )
    
    #setup scheduler
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    
    
    #Build engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    #start_epoch = torchreid.utils.resume_from_checkpoint(
     #   'log/osnet/model/osnet_x1_0_market.pth',
      #  model,
       # optimizer
    #)        
   
    #run training and test
    engine.run(
        save_dir='log/osnet',
        max_epoch=10, #make start_epoch when loading and testing
        eval_freq=10,
        print_freq=50,
        visrank = True,
        visrank_topk=2,
        test_only=True
    )
    
if __name__ == "__main__":
    main()