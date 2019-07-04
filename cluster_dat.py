#!/usr/bin/env python3

import argparse
import logging

import numpy as np
import pandas as pd
import pylab as plt
from sklearn.cluster import DBSCAN

__author__ = 'Devansh Agarwal'
__email__ = 'da0017@mix.wvu.edu'

TSAMP = 327.68e-6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DBSCAN on the .dat files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-t', '--tdist', type=int, help='Sample distance (in number of samples)', default=16)
    parser.add_argument('-d', '--ddist', type=int, help='DM distance (in pc/cc)', default=1)
    parser.add_argument('-p', '--plot', action='store_true', help='Show plots', default=False)
    parser.add_argument('-w', '--log_width', type=int, help='Log width distance', default=1)
    parser.add_argument('-m', '--minsamp', type=int,
                        help='Minimum samples in cluster (Use m=1 for friends of friends)',
                        default=1)
    parser.add_argument('--njobs', type=int, help='The number of parallel jobs to run (-1 for all cores)', default=1)
    parser.add_argument(dest='file', help="eg Beam2_dm_D20160429T233418.dat")

    args = parser.parse_args()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    with open(args.file, 'r') as ff:
        content = ff.readlines()

    time_step = args.tdist
    dm_step = args.ddist
    width_step = args.log_width
    min_samples = args.minsamp
    n_jobs = args.njobs

    list_of_buffers = []
    MJD_list = []
    temp_list = []
    for line in content:
        line = line.split()
        if "#" in line and "Written" in line:
            buffer_id = int(line[3])
            MJD = float(line[6])
            list_of_buffers.append(temp_list)
            MJD_list.append(MJD)
            temp_list = []
        elif "#" not in line:
            temp_list.append(np.array([x.strip(',') for x in line], dtype='f8'))

    logging.info(f'Read {len(list_of_buffers)} blocks')
    centers = []

    for block_id, each_list in enumerate(list_of_buffers):
        df = pd.DataFrame(each_list, columns=['Time', 'DM', 'SNR', 'Width'], index=None)
        df['MJD_file'] = np.array(MJD_list[block_id], dtype=np.float64)
        df['tcand'] = (df['Time'] - df['MJD_file']) * 24 * 3600
        df['log_width'] = np.log2(df['Width'])

        data = np.array((df['tcand'] * TSAMP, df['DM'] / dm_step, df['log_width'] / width_step),
                        dtype=np.float32).T

        db = DBSCAN(eps=np.sqrt(3), min_samples=min_samples, n_jobs=n_jobs).fit(data)

        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        unique_labels = set(labels)
        colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, len(unique_labels))]

        if args.plot:
            plt.figure(figsize=(9, 7))
            plt.clf()

        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            cluster_mask = class_member_mask & core_samples_mask
            max_snr_cand_id = df[cluster_mask]['SNR'].idxmax()
            row = df.iloc[max_snr_cand_id]
            row['members'] = cluster_mask.sum()
            row['block_id'] = block_id
            centers.append(row)
            if args.plot:
                plt.scatter(df['tcand'][cluster_mask], df['DM'][cluster_mask], color=tuple(col), alpha=0.1)
                plt.scatter(df['tcand'].iloc[max_snr_cand_id], df['DM'].iloc[max_snr_cand_id], color=tuple(col),
                            s=10 * df['SNR'].iloc[max_snr_cand_id])

        if args.plot:
            plt.xlabel('Time (s)')
            plt.ylabel('DM (pc/cc)')
            plt.title(f'Block ID: {block_id}, No. of clusters: {len(unique_labels)} ')
            plt.savefig(f'{block_id}.png', bbox_inches='tight')

        logging.info(f'Block id: {block_id} Reduced from {len(df)} to {len(unique_labels)} candidates')

    clustered_cands = pd.DataFrame(centers)

    dbs_file_name = args.file[:-4] + '.dbs'

    clustered_cands.to_csv(dbs_file_name, index=None)
