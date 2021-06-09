import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

# In order to change the format following FairMOT
# You need to change seq_root to your training set
seq_root = '/content/drive/MyDrive/William_house/Cam0'
# Also change the label_root to your save path
label_root = '/content/drive/MyDrive/William_house/labels_with_ids'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_width = 1920
    seq_height = 1080

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    print(seq)
    seq_label_root = osp.join(label_root, seq, 'im')
    mkdirs(seq_label_root)
    file_dir = osp.join(seq_root, seq, 'im')
    # change this path for save your ourdataset.train
    txt_path = osp.join('/content/FairMOT/src/data', 'ourdataset.train')
    for fid, tid, x, y, w, h, label, mark, _, r in gt:
        if mark == 0 or not label == 1:
          continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        file_path = '{:06d}.jpg'.format(fid)
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
        with open(txt_path, 'a') as fi:
          fi.write(osp.join(file_dir,file_path)+'\n')