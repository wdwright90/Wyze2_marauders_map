import sys
import os
import glob
import re
import os.path as osp

import numpy as np
sys.path.append("/content/Wyze2_marauders_map/REID_model/deep-person-reid")
import torch

from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix

"""A simple API for feature extraction.

FeatureExtractor can be used like a python function, which
accepts input of the following types:
    - a list of strings (image paths)
    - a list of numpy.ndarray each with shape (H, W, C)
    - a single string (image path)
    - a single numpy.ndarray with shape (H, W, C)
    - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

Returned is a torch tensor with shape (B, D) where D is the
feature dimension.

Args:
    model_name (str): model name.
    model_path (str): path to model weights.
    image_size (sequence or int): image height and width.
    pixel_mean (list): pixel mean for normalization.
    pixel_std (list): pixel std for normalization.
    pixel_norm (bool): whether to normalize pixels.
    device (str): 'cpu' or 'cuda' (could be specific gpu devices).
    verbose (bool): show model details.

Examples::

    from torchreid.utils import FeatureExtractor

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='a/b/c/model.pth.tar',
        device='cuda'
    )

    image_list = [
        'a/b/c/image001.jpg',
        'a/b/c/image002.jpg',
        'a/b/c/image003.jpg',
        'a/b/c/image004.jpg',
        'a/b/c/image005.jpg'
    ]

    features = extractor(image_list)
    print(features.shape) # output (5, 512)
"""
#current_cam is which cams data is being input
def eval_cam(image_dict, current_cam):
  new_pid_threshold_same_cam = 10
  new_pid_threshold_diff_cam = 25
  matched_ids = []#This is our return vector which will hold all our matched ids
  extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='/content/Wyze2_marauders_map/deep_sort/deep_sort/checkpoint/model.pth.tar-10',
    device='cuda'
  )

  #First we check if the saved features dir exists and create it otherwise
  if not os.path.isdir('/content/Wyze2_marauders_map/saved_features'):
    os.mkdir('/content/Wyze2_marauders_map/saved_features')

  #Could be this image list but also could just be
  image_list = []
  for identity in image_dict:
    print(identity)
    matched_id = (identity, identity) #matched identity tuple, deepsort at pos 0 and reid at pos 1
    features = extractor(image_dict[identity])
    qf = torch.mean(features, 0)
    #print("features after mean")
    #print(qf.shape) # output (5, 512)

    #Next we check if the saved features directory is empty
    saved_features_vectors = glob.glob(osp.join('/content/Wyze2_marauders_map/saved_features', '*.pt'))
    #print("my saved_features_vectors")
    #print(saved_features_vectors)
    num_saved_features_vectors = len(saved_features_vectors)
    save_location_and_name = osp.join('/content/Wyze2_marauders_map/saved_features', 'features_pid{}_cam0_track{}.pt'.format(identity, num_saved_features_vectors))
    if num_saved_features_vectors < 1:
      #This means saved features directory is empty, save our current feature vector
      torch.save(qf, save_location_and_name)
    else:
      #This means there are already saved feature vectors we should compare against
      closest_pid_diff_cam = -1
      closest_pid_same_cam = -1
      smallest_dist_diff_cam = 10000 #making this big enough just to check if things are smaller
      smallest_dist_same_cam = 10000
      for feature_vector in saved_features_vectors:
        gf = torch.load(feature_vector)
        dist = torch.dist(qf, gf, p=2)
        #print("the distance i got was:{}".format(dist))
        #must parse the feature_vector loaded string to get which pid it corresponds to
        path_list = feature_vector.split(os.sep)
        #print(path_list)
        featurevec_name = path_list[-1]
        #print(featurevec_name)
        features, pid, camid, track_num = featurevec_name.split("_")
        #print(pid)
        pid_num = int(re.findall(r'\d+', pid)[0])
        #print("pid_num is {}".format(pid_num))
        camid_num = int(re.findall(r'\d+', camid)[0])
        #print("camid_num is {}".format(camid_num))
        if (dist < smallest_dist_diff_cam and current_cam != camid_num):
          smallest_dist_diff_cam = dist
          closest_pid_diff_cam = pid_num
        if (dist < smallest_dist_same_cam and current_cam == camid_num):
          smallest_dist_same_cam = dist
          closest_pid_same_cam = pid_num
      if (smallest_dist_diff_cam > new_pid_threshold_diff_cam and smallest_dist_same_cam > new_pid_threshold_same_cam):
        #This must be a new identity
        print("Created a new identity, both thresholds were exceeded")
        torch.save(qf, save_location_and_name)
      elif (smallest_dist_diff_cam < new_pid_threshold_diff_cam):
        #This must be a new identity
        print("matched to a different cam")
        save_location_and_name = osp.join('/content/Wyze2_marauders_map/saved_features', 'features_pid{}_cam0_track{}.pt'.format(closest_pid_diff_cam, num_saved_features_vectors))
        torch.save(qf, save_location_and_name)
        matched_id = (identity, closest_pid_diff_cam)
      else:
        printed("hit the else, matched to closest cam")
        #This is a matching identity, save it with the matched id instead of deepsort id
        save_location_and_name = osp.join('/content/Wyze2_marauders_map/saved_features', 'features_pid{}_cam0_track{}.pt'.format(closest_pid_same_cam, num_saved_features_vectors))
        torch.save(qf, save_location_and_name)
        matched_id = (identity, closest_pid_same_cam) #In this case we must change the matched id

    matched_ids.append(matched_id) #append the matched identity to our output list
  #Finally we return the matched ids for usage by the 
  return matched_ids
