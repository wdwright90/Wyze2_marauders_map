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

from torchreid.data import VideoDataset



class Wyze2_video(VideoDataset):
    dataset_dir = 'wyze2_video/tracks'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.camera_count = 4

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
        
        self.train_dir = self.dataset_dir
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.dataset_dir
        
        #   in query/gallery).
        train = self.process_dir(self.train_dir, 'train')
        query = self.process_dir(self.query_dir, 'query')
        gallery = self.process_dir(self.gallery_dir, 'gallery')

        super(Wyze2_video, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path, train_guery_gallery):
        dir_names = os.listdir(dir_path)
        data = []
        for name in dir_names:
            if name == 'ilids':
                ilids_dir_path = osp.join(dir_path, 'ilids')
                ilids_dir_names = os.listdir(ilids_dir_path)
                for person_folder in ilids_dir_names:
                    cur_path = osp.join(ilids_dir_path, person_folder)
                    img_paths = glob.glob(osp.join(cur_path, '*.png'))
                    cam_0_paths = []
                    cam_1_paths = []
                    path_list = img_paths[0].split(os.sep)
                    img_name = path_list[-1]
                    camid, pid, img_num = img_name.split("_")
                    
                    pid_num = int(re.findall(r'\d+', pid)[0])
                    for img_path in img_paths:
                        path_list = img_path.split(os.sep)
                        img_name = path_list[-1]
                        camid, pid, img_num = img_name.split("_")
                        
                        pid_num = int(re.findall(r'\d+', pid)[0])
                        camid_num = int(re.findall(r'\d+', camid)[0]) - 1 #Subtract 1 because ilids uses 1 indexing
                        if (camid_num == 0):
                            cam_0_paths.append(img_path)
                        else:
                            cam_1_paths.append(img_path)
                            
                    if (train_guery_gallery == 'query'):
                        data.append((cam_0_paths, pid_num, 0))
                    elif (train_guery_gallery == 'gallery'):
                        data.append((cam_1_paths, pid_num, 1))
                    elif (train_guery_gallery == 'train'):
                        data.append((cam_0_paths, pid_num, 0))
                        data.append((cam_1_paths, pid_num, 1))
                    else:
                        pass
            elif name == 'wyze2':
                wyze2_dir_path = osp.join(dir_path, 'wyze2')
                wyze2_dir_names = os.listdir(wyze2_dir_path)
                for person_folder in wyze2_dir_names:
                    cur_path = osp.join(wyze2_dir_path, person_folder)
                    img_paths = []
                    for ext in ('*.jpg', '*.png'):
                        img_paths.extend(glob.glob(osp.join(cur_path, ext)))
                    path_list = img_paths[0].split(os.sep)
                    img_name = path_list[-1]
                    pid, camid, trackid, img_num = img_name.split("_")
                    
                    pid_num = int(re.findall(r'\d+', pid)[0])
                    cam_range = range(self.camera_count)
                    if (train_guery_gallery == 'query'):
                        cam_range = range(1)
                    elif (train_guery_gallery == 'gallery'):
                        cam_range = range(1, self.camera_count)
                    
                    for i in cam_range:
                        same_cam_paths = []
                        for img_path in img_paths:
                            
                            path_list = img_path.split(os.sep)
                            img_name = path_list[-1]
                            pid, camid, trackid, img_num = img_name.split("_")
                            
                            pid_num = int(re.findall(r'\d+', pid)[0])
                            camid_num = int(re.findall(r'\d+', camid)[0])
                            if (camid_num == i):
                                same_cam_paths.append(img_path)
                            else:
                                pass
                        if (len(same_cam_paths) > 10):
                            data.append((same_cam_paths, pid_num, i))
                
                #print(data)
            
        return data
    


import torchreid


def main ():
    #register for videos instead
    torchreid.data.register_video_dataset('wyze2_video', Wyze2_video)
    
    #Load video datamanager
    datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources= 'wyze2_video',
            height=256,
            width=128,
            batch_size_train=3,
            batch_size_test=3,
            seq_len=10,
            sample_method='evenly'
        )
    
    #Build model
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=150, #originally was this but to load a checkpoint we need to match the trained model class count datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
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
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    start_epoch = torchreid.utils.resume_from_checkpoint(
        'log/osnet/model/model.pth.tar-10',
        model,
        optimizer
    )        
   
    #run training and test
    engine.run(
        save_dir='log/osnet',
        max_epoch=10, #make start_epoch when loading and testing
        eval_freq=10,
        print_freq=50,
        visrank = False,
        visrank_topk=2,
        test_only=True
    )
    
if __name__ == "__main__":
    main()