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
    def __init__(self, args,vpath,spath):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        self.vpath = vpath
        self.spath = spath
        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.vpath), "Error: path error"
        self.vdo.open(self.vpath)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.spath:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.spath, fourcc, 15, (self.im_width, self.im_height))

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
        cam_num = int(re.findall(r'\d+', self.vpath)[1])
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
            if self.spath:
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
    parser.add_argument("--save_path", type=str, default="/content/")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spath = []
    vpath = args.VIDEO_PATH.split('_next')
    n = len(vpath)
    for i in range(n):
        spath.append(os.path.join(args.save_path,'example{}.avi'.format(i)))
    for i in range(n):
        with Detector(args,vpath[i],spath[i]) as det:
            det.detect()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    newpath = os.path.join(args.save_path,'videos.avi')
    print('Integrating {} videos processing...'.format(n))
    if n == 1:
        print('Only one videos')
    elif n == 2:
        videoLeft = cv2.VideoCapture(spath[0])
        videoRight = cv2.VideoCapture(spath[1])
        fps = videoLeft.get(cv2.CAP_PROP_FPS)
        width = (int(videoLeft.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = (int(videoLeft.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videoWriter = cv2.VideoWriter(newpath, fourcc, fps, (width, height))

        successLeft, frameLeft = videoLeft.read()
        successRight, frameRight = videoRight.read()
        target = np.zeros((height,width),dtype=np.uint8)

        frame = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        while successLeft and successRight:
            frameLeft = cv2.resize(frameLeft, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameRight = cv2.resize(frameRight, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            image = np.hstack((frameLeft, frameRight))
            h = image.shape[0]
            w = image.shape[1]

            for i in range(h/2):
                for j in range(w):
                    frame[i, j, 0] = image[i, j, 0]
                    frame[i, j, 1] = image[i, j, 1]
                    frame[i, j, 2] = image[i, j, 2]

            videoWriter.write(frame)
            successLeft, frameLeft = videoLeft.read()
            successRight, frameRight = videoRight.read()

        videoWriter.release()
        videoLeft.release()
        videoRight.release()
    elif n == 3:
        videoLeftUp = cv2.VideoCapture(spath[0])
        videoLeftDown = cv2.VideoCapture(spath[1])
        videoRightUp = cv2.VideoCapture(spath[2])
        fps = videoLeftUp.get(cv2.CAP_PROP_FPS)
        width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videoWriter = cv2.VideoWriter(newpath, fourcc, fps, (width, height))

        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown , frameLeftDown = videoLeftDown.read()
        successRightUp, frameRightUp = videoRightUp.read()

        target = np.zeros((height,width),dtype=np.uint8)

        frame = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        while successLeftUp and successLeftDown and successRightUp:
            frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

            imageup = np.hstack((frameLeftUp, frameRightUp))
            h = imageup.shape[0]
            w = imageup.shape[1]

            for i in range(h):
                for j in range(w):

                    frame[i, j, 0] = imageup[i, j, 0]
                    frame[i, j, 1] = imageup[i, j, 1]
                    frame[i, j, 2] = imageup[i, j, 2]
            for i in range(h,2*h):
                for j in range(int(w/2)):
                    frame[i, j, 0] = frameLeftDown[i-h, j, 0]
                    frame[i, j, 1] = frameLeftDown[i-h, j, 1]
                    frame[i, j, 2] = frameLeftDown[i-h, j, 2]

            videoWriter.write(frame)
            successLeftUp, frameLeftUp = videoLeftUp.read()
            successLeftDown, frameLeftDown = videoLeftDown.read()
            successRightUp, frameRightUp = videoRightUp.read()

        videoWriter.release()
        videoLeftUp.release()
        videoLeftDown.release()
        videoRightUp.release()
    elif n == 4:
        videoLeftUp = cv2.VideoCapture(spath[0])
        videoLeftDown = cv2.VideoCapture(spath[2])
        videoRightUp = cv2.VideoCapture(spath[1])
        videoRightDown = cv2.VideoCapture(spath[3])
        fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

        width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videoWriter = cv2.VideoWriter(newpath, fourcc, fps, (width, height))

        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown , frameLeftDown = videoLeftDown.read()
        successRightUp, frameRightUp = videoRightUp.read()
        successRightDown, frameRightDown = videoRightDown.read()

        while successLeftUp and successLeftDown and successRightUp and successRightDown:
            frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
            frameRightDown = cv2.resize(frameRightDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

            frameUp = np.hstack((frameLeftUp, frameRightUp))
            frameDown = np.hstack((frameLeftDown, frameRightDown))
            frame = np.vstack((frameUp, frameDown))

            videoWriter.write(frame)
            successLeftUp, frameLeftUp = videoLeftUp.read()
            successLeftDown, frameLeftDown = videoLeftDown.read()
            successRightUp, frameRightUp = videoRightUp.read()
            successRightDown, frameRightDown = videoRightDown.read()

        videoWriter.release()
        videoLeftUp.release()
        videoLeftDown.release()
        videoRightUp.release()
        videoRightDown.release()
    else:
        print('Too many videos!')