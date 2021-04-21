import os
import os.path as osp
import logging
import argparse
from pathlib import Path
import numpy as np
from utils.log import get_logger
from deepsort_reid import Detector
import motmetrics as mm

mm.lap.default_solver = 'lap'
from utils.evaluation import Evaluator


def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)


def main(data_root='', cams=('',), seqs=('',), args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    result_root = osp.join(Path(data_root), "mot_results")
    mkdir_if_missing(result_root)

    # run tracking
    accs = []
    for cam in cams:
        for seq in seqs:
            logger.info('start cam: {} seq: {}'.format(cam, seq))
            result_filename = osp.join(result_root, '{}_{}.txt'.format(cam, seq))

            video_path = data_root + '/' + cam + '/' + seq + '/{}_{}.mp4'.format(cam, seq)
            spath = os.path.join(args.save_path, 'video{}_{}.avi'.format(cam, seq))

            with Detector(args, video_path, spath, result_filename) as det:
                det.detect(cam=cam, seq=seq)

            # eval
            logger.info('Evaluate cam: {} seq: {}'.format(cam, seq))
            evaluator = Evaluator(data_root, cam, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    name = []
    for i in range(len(cams)):
        for j in range(len(seqs)):
            name.append(cams[i] + '_' + seqs[j])
    name = np.array(name)
    summary = Evaluator.get_summary(accs, name, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/content/Wyze2_marauders_map/deep_sort/deep_sort/checkpoint/new3ckpt_30epochs.t7')
    parser.add_argument('--save_path', type=str, default='/content/Wyze2_marauders_map/evaluation')
    parser.add_argument('--use_cuda', type=str, default='True')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cams_str = '''
                Cam0
                Cam1
                Cam2
                Cam3
                

                '''
    seqs_str = '''      
                  Seq0
                  Seq1
                  Seq2
                  Seq3
                  Seq4

                  '''
    data_root = '/content/Wyze2_marauders_map/data/VideoData'
    cams = [cams.strip() for cams in cams_str.split()]
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root, cams=cams, seqs=seqs, args=args)
