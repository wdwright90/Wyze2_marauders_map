
import argparse
import os
import time
from distutils.util import strtobool
import numpy as np
import re

import cv2
import my_eval_script

from Deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def reid():
        pass

    def detect(self):
        count = 0
        store_im = []
        store_bbox = []
        store_ids = []
        store_out = []
        identities_and_images = {}
        cam_num = int(re.findall(r'\d+', self.args.VIDEO_PATH)[1])
        while self.vdo.grab():
            _, im = self.vdo.retrieve()
            store_im.append(im)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)
            if len(bbox_xcycwh) is 0:
                store_bbox.append(0)
                store_ids.append(0)
            else:
                store_bbox.append(bbox_xcycwh)
                store_ids.append(cls_ids)
            if len(bbox_xcycwh) is not 0:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                store_out.append(outputs)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for i in range(len(identities)):
                        x1,y1,x2,y2 = bbox_xyxy[i]
                        im_crop = im[y1:y2,x1:x2]
                        im_crop = np.asarray(im_crop)
                        #cv2.imwrite('/content/drive/MyDrive/EE597/result/cam1_seq0_image/cam0_seq0_id{}'.format(identities[i]) + '_{}.png'.format(count), im_crop)
                        if identities[i] in identities_and_images:
                            identities_and_images[identities[i]].append(im_crop)
                        else:
                            identities_and_images[identities[i]] = [im_crop]
        matched_ids = my_eval_script.eval_cam(identities_and_images, cam_num)
        print(matched_ids)
        # loop for generate video with matched_ids
        for i in range(len(store_im)):
            bbox_xcycwh = store_bbox[i]
            im = store_im[i]
            cls_ids = store_ids[i]
            if bbox_xcycwh is not 0:
                # select class person
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                outputs = store_out[count]
                count += 1
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for i in range(len(identities)): #deepsort tracker id
                        for old_id, new_id in matched_ids: # matched_ids part list
                            if old_id == identities[i]:
                                identities[i] = new_id
                    im = draw_bboxes(im, bbox_xyxy, identities)
            if self.args.save_path:
                self.output.write(im)
        #Write images only when finished

        #if self.args.save_path:
        #    im = draw_bboxes(im, bbox_xyxy, identities)
        #    self.output.write(im)
        #        count += 1
        #        im = draw_bboxes(im, bbox_xyxy, identities)
        #
        #    end = time.time()
        #    print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
        #    if self.args.save_path:
        #        self.output.write(im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="/content/Wyze2_marauders_map/deep_sort/deep_sort/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()