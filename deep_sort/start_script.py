# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:32:16 2021

@author: William
"""
import deep_sort_app

"""

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

"""
    
sequence_dir = "./MOT16/test/MOT16-06"
detection_file = None #"./resources/detections/MOT16_POI_test/MOT16-06.npy"
output_file = "/tmp/hypotheses.txt"
min_confidence = 0.3
nms_max_overlap = 1.0
min_detection_height = 0
max_cosine_distance = 0.2
nn_budget = 100
display = True


deep_sort_app.run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display)