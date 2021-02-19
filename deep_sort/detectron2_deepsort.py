import argparse
import os
import time
from distutils.util import strtobool

import cv2

from .Deep_sort import DeepSort
from .detectron2_detection import Detectron2
from .util import draw_bboxes

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

    def detect(self):
        count = 0
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            if len(bbox_xcycwh) is not 0:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for i in range(len(identities)):
                        x1,y1,x2,y2 = bbox_xyxy[i]
                        im_crop = im[y1:y2,x1:x2]
                        cv2.imwrite('/content/drive/MyDrive/EE597/result/cam1_seq0_image/cam0_seq0_id{}'.format(identities[i]) + '_{}.png'.format(count), im_crop)
                        count += 1
                    im = draw_bboxes(im, bbox_xyxy, identities)

            end = time.time()
            print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
            if self.args.save_path:
                self.output.write(im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="Wyze2_marauders_map\deep_sort\deep_sort\checkpoint\ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.MP4")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()