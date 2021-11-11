# encoding=utf-8

"""
Created by Xiao Aoran @ntu, sg. 2020/Jun/23 10:38 am.
"""

import argparse
import os
import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
# from sklearn.metrics import adjusted_rand_score
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
import pdb

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def read_scan_from_infer(infer_scan_path):
    origin_scan_path = os.path.join('/home_nfs/aoran.xiao/projects/datasets/cloudpoint/SemanticKITTI/dataset/sequences',
                                    infer_scan_path.split('/')[-3], 'velodyne', infer_scan_path.split('/')[-1].replace('label', 'bin'))
    assert os.path.exists(origin_scan_path)
    scan = np.fromfile(origin_scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-l', type=str, required=True, help="path for logdir")
    args = parser.parse_args()

    # args.arch_cfg = os.path.join(args.logdir, 'arch_cfg.yaml')
    # args.data_cfg = os.path.join(args.logdir, 'data_cfg.yaml')
    #
    # # open arch config file
    # try:
    #     print("Opening arch config file %s" % args.arch_cfg)
    #     ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    # except Exception as e:
    #     print(e)
    #     print("Error opening arch yaml file.")
    #     quit()
    #
    # # open data config file
    # try:
    #     print("Opening data config file %s" % args.data_cfg)
    #     DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    # except Exception as e:
    #     print(e)
    #     print("Error opening data yaml file.")
    #     quit()

    # get infer files list
    infer_dir = os.path.join(args.logdir, 'infer', 'sequences', '08')
    infer_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(infer_dir)) for f in fn if is_label(f)]
    for index in range(len(infer_files)):
        # get infer data
        infer_scan = np.fromfile(infer_files[index], dtype=np.int32)
        # get corresponding scan data for xyz
        origin_scan = read_scan_from_infer(infer_files[index])

        object_class = 10  # 'car'
        object_points_indexes = np.where(infer_scan == object_class)[0]
        object_points = origin_scan[object_points_indexes][: , :3]

        #  ************* cluster *************
        # 1. hierarchical not suitable, k need to be set
        # linkage = 'average'  # ['average','complete']
        # clu = AgglomerativeClustering(n_clusters=2, linkage=linkage)  # n_clusters:the number of clusters we want;linkage:the way to calculate the distance
        # cluster_output = clu.fit_predict(object_points)

        # 2. Mean shift
        pdb.set_trace()
        bandwidth = estimate_bandwidth(object_points, quantile=0.2, n_samples=1000)
        ms = MeanShift(bandwidth=bandwidth, n_jobs=10, bin_seeding=True)  #
        ms.fit(object_points)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)




