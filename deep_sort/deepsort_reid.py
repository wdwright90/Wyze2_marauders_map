import argparse
import os
import time
from distutils.util import strtobool
import cv2
from Deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes
from utils.log import get_logger
from utils.io import write_results
import numpy as np
import my_eval_script

class Detector(object):
    def __init__(self, args, video_path, result_filename):
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.result_filename = result_filename
        use_cuda = bool(strtobool(self.args.use_cuda))

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Error: path error"
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 15, (self.im_width, self.im_height))

            self.logger.info("Save results to {}".format(self.args.save_path))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self, cam=None):
        count = 0
        results = []
        idx_frame = 0
        store_im = []
        store_bbox = []
        store_ids = []
        store_out = []
        identities_and_images = {}
        cam_num = int(str(cam)[-1])
        while self.vdo.grab():
            start = time.time()
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
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im, cam)
                store_out.append(outputs)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for i in range(len(identities)):
                        x1, y1, x2, y2 = bbox_xyxy[i]
                        im_crop = im[y1:y2, x1:x2]
                        im_crop = np.asarray(im_crop)
                        if identities[i] in identities_and_images:
                            identities_and_images[identities[i]].append(im_crop)
                        else:
                            identities_and_images[identities[i]] = [im_crop]
                    end = time.time()
                    self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                                     .format(end - start, 1 / (end - start), bbox_xcycwh.shape[0], len(outputs)))
        matched_ids = my_eval_script.eval_cam(identities_and_images, cam_num)
        print(matched_ids)

        for i in range(len(store_im)):
            bbox_xcycwh = store_bbox[i]
            im = store_im[i]
            cls_ids = store_ids[i]
            idx_frame += 1

            if bbox_xcycwh is not 0:
                # select class person
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                outputs = store_out[count]
                count += 1
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for i in range(len(identities)):  # deepsort tracker id
                        for old_id, new_id in matched_ids:  # matched_ids part list
                            if old_id == identities[i]:
                                identities[i] = new_id
                    im = draw_bboxes(im, bbox_xyxy, identities)
                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                    results.append((idx_frame - 1, bbox_tlwh, identities))

                if self.args.save_path:
                    self.output.write(im)
                write_results(self.result_filename, results, 'mot')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--checkpoint", type=str,
                        default="/content/Wyze2_marauders_map/deep_sort/deep_sort/checkpoint/ckpt.t7")
    parser.add_argument("--save_path", type=str, default="/content/")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_filename = '/content/'
    with Detector(args=args, video_path=args.VIDEO_PATH, result_filename=result_filename) as det:
        det.detect()