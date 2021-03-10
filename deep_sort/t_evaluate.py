import os
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from detectron2_deepsort import Detector
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
            logger.info('start cam: {}, seq: {}'.format(cam,seq))
            result_filename = osp.join(result_root, '{}_{}.txt'.format(cam, seq))
            video_path = []
            # for i in range(len(os.listdir(osp.join(data_root, cam, seq)))-1):
            #     video_path.append(data_root + "/" + cam + "/" + seq + "{}_{}_".fotmat(i))

            with Detector(args, video_path) as det:
                det.detect()

            # eval
            logger.info('Evaluate cam: {}, seq: {}'.format(cam, seq))
            evaluator = Evaluator(data_root, cam, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, cams, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="/content/demo.avi")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cams_str = '''
                Cam0
                Cam1
                Cam2
                Cam3
                Cam4
                Cam5
                '''
    seqs_str = '''Seq0      
                  Seq1
                  Seq2
                  Seq3
                  Seq4
                  '''
    data_root = '/content/Wyze2_marauders_map/data/VideoData'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs,
         args=args)
