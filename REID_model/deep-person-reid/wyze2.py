# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:07:04 2021

@author: William
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from torchreid.data import ImageDataset


class wyze2(ImageDataset):
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
        
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        
        #   in query/gallery).
        train = process_dir(self.train_dir)
        query = process_dir(self.query_dir)
        gallery = process_dir(self.gallery_dir)

        super(wyze2, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path):
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        print("variable img_paths is:")
        print(img_paths)