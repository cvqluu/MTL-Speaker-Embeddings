import argparse
import concurrent.futures
import configparser
import glob
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvloop
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_io import DiarizationDataset
from models.extractors import ETDNN, FTDNN, XTDNN
from train import parse_config


def parse_args():
    parser = argparse.ArgumentParser(description='Diarize per recording')
    parser.add_argument('--cfg', type=str,
                        default='./configs/example_speaker.cfg')
    parser.add_argument('--checkpoint', type=int,
                        default=0, help='Choose a specific iteration to evaluate on instead of the best eer')
    parser.add_argument('--diar-data', type=str,
                        default='/disk/scratch2/s1786813/repos/supreme_court_transcripts/oyez/scotus_diarization_nosil/test')
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args


def lines_to_file(lines, filename, wmode="w+"):
    with open(filename, wmode) as fp:
        for line in lines:
            fp.write(line)


def get_eer_metrics(folder):
    rpkl_path = os.path.join(folder, 'results.p')
    rpkl = pickle.load(open(rpkl_path, 'rb'))
    iterations = list(rpkl.keys())
    eers = [rpkl[k]['test_eer'] for k in rpkl]
    return iterations, eers, np.min(eers)


def setup():
    if args.model_type == 'XTDNN':
        generator = XTDNN(features_per_frame=args.input_dim,
                          embed_features=args.embedding_dim)
    if args.model_type == 'ETDNN':
        generator = ETDNN(features_per_frame=args.input_dim,
                          embed_features=args.embedding_dim)
    if args.model_type == 'FTDNN':
        generator = FTDNN(in_dim=args.input_dim,
                          embedding_dim=args.embedding_dim)

    generator.eval()
    generator = generator
    return generator


def agg_clustering_oracle(S, num_clusters):
    ahc = AgglomerativeClustering(
        n_clusters=num_clusters, affinity='precomputed', linkage='average', compute_full_tree=True)
    return ahc.fit_predict(S)


def score_der(hyp=None, ref=None, outfile=None, collar=0.25):
    '''
    Takes in hypothesis rttm and reference rttm and returns the diarization error rate
    Calls md-eval.pl -> writes output to file -> greps for DER value
    '''
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref)
    assert outfile
    cmd = 'perl md-eval.pl -1 -c {} -s {} -r {} > {}'.format(
        collar, hyp, ref, outfile)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    assert os.path.isfile(outfile)
    with open(outfile, 'r') as file:
        data = file.read().replace('\n', '')
        der_str = re.search(
            r'DIARIZATION\ ERROR\ =\ [0-9]+([.][0-9]+)?', data).group()
    der = float(der_str.split()[-1])
    return der


def score_der_uem(hyp=None, ref=None, outfile=None, uem=None, collar=0.25):
    '''
    takes in hypothesis rttm and reference rttm and returns the diarization error rate
    calls md-eval.pl -> writes output to file -> greps for der value
    '''
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref)
    assert os.path.isfile(uem)
    assert outfile
    cmd = 'perl md-eval.pl -1 -c {} -s {} -r {} -u {} > {}'.format(
        collar, hyp, ref, uem, outfile)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    assert os.path.isfile(outfile)
    with open(outfile, 'r') as file:
        data = file.read().replace('\n', '')
        der_str = re.search(
            r'DIARIZATION\ ERROR\ =\ [0-9]+([.][0-9]+)?', data).group()
    der = float(der_str.split()[-1])
    return der

def make_rttm_lines(segcols, cluster_labels):
    # Make the rtttm from segments and cluster labels, resolve overlaps etc
    assert len(segcols[0]) == len(cluster_labels)
    assert len(set(segcols[1])) == 1, 'Must be from a single recording'
    rec_id = list(set(segcols[1]))[0]

    starts = segcols[2].astype(float)
    ends = segcols[3].astype(float)
    
    events = [{'start': starts[0], 'end': ends[0], 'label': cluster_labels[0]}]
    
    for t0, t1, lab in zip(starts, ends, cluster_labels):
        # TODO: Warning this only considers overlap with a single adjacent neighbour
        if t0 <= events[-1]['end']:
            if lab == events[-1]['label']:
                events[-1]['end'] = t1
                continue
            else:
                overlap = events[-1]['end'] - t0
                events[-1]['end'] -= overlap/2
                newevent = {'start': t0+overlap/2, 'end': t1, 'label': lab}
                events.append(newevent)
        else:
            newevent = {'start': t0, 'end': t1, 'label': lab}
            events.append(newevent)
    
    line_str = 'SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n'
    lines = []
    for ev in events:
        offset = ev['end'] - ev['start']
        if offset < 0.0:
            continue
        lines.append(line_str.format(rec_id, ev['start'], offset, ev['label']))
    return lines


def extract_and_diarize(generator, test_data_dir, device):
    all_hyp_rttms = []
    all_ref_rttms = []

    # if args.checkpoint == 0:
    #     results_pkl = os.path.join(args.model_dir, 'diar_results.p')
    # else:
    results_pkl = os.path.join(args.model_dir, 'diar_results_{}.p'.format(args.checkpoint))
    rttm_dir = os.path.join(args.model_dir, 'hyp_rttms')
    os.makedirs(rttm_dir, exist_ok=True)

    if os.path.isfile(results_pkl):
        rpkl = pickle.load(open(results_pkl, 'rb'))
        if rpkl['test_data'] != TEST_DATA_PATH:
            moved_rpkl = os.path.join(
                args.model_dir, rpkl['test_data'].replace(os.sep, '-') + '.p')
            shutil.copy(results_pkl, moved_rpkl)
            rpkl = OrderedDict({'test_data': TEST_DATA_PATH})
    else:
        rpkl = OrderedDict({'test_data': TEST_DATA_PATH})
    
    if 'full_der' in rpkl:
        print('({}) Full test DER: {}'.format(args.cfg, rpkl['full_der']))
        if 'full_der_uem' not in rpkl:
            uem_file = os.path.join(TEST_DATA_PATH, 'uem')
            der_uem = score_der_uem(hyp=os.path.join(args.model_dir, 'final_{}_hyp.rttm'.format(args.checkpoint)),
                                    ref=os.path.join(args.model_dir, 'final_{}_ref.rttm'.format(args.checkpoint)),
                                    outfile=os.path.join(args.model_dir, 'final_hyp_uem.derlog'),
                                    uem=uem_file,
                                    collar=0.25)
            rpkl['full_der_uem'] = der_uem
            pickle.dump(rpkl, open(results_pkl, 'wb'))
        print('({}) Full test DER (uem): {}'.format(args.cfg, rpkl['full_der_uem']))
        
    else:
        ds_test = DiarizationDataset(test_data_dir)
        recs = ds_test.recs
        generator = setup()
        generator.eval().to(device)

        if args.checkpoint == 0:
            generator_its, eers, _ = get_eer_metrics(args.model_dir)
            g_path = os.path.join(args.model_dir, 'generator_{}.pt'.format(
                generator_its[np.argmin(eers)]))
        else:
            g_path = os.path.join(args.model_dir, 'generator_{}.pt'.format(args.checkpoint))
            assert os.path.isfile(g_path), "Couldn't find {}".format(g_path)

        generator.load_state_dict(torch.load(g_path))

        with torch.no_grad():
            for i, r in tqdm(enumerate(recs), total=len(recs)):
                ref_rec_rttm = os.path.join(rttm_dir, '{}_{}_ref.rttm'.format(r, args.checkpoint))
                hyp_rec_rttm = os.path.join(rttm_dir, '{}_{}_hyp.rttm'.format(r, args.checkpoint))
                if r in rpkl and (os.path.isfile(ref_rec_rttm) and os.path.isfile(hyp_rec_rttm)):
                    all_ref_rttms.append(ref_rec_rttm)
                    all_hyp_rttms.append(hyp_rec_rttm)
                    continue

                feats, spkrs, ref_rttm_lines, segcols, rec = ds_test.__getitem__(i)
                num_spkrs = len(set(spkrs))
                assert r == rec

                # Extract embeds
                embeds = []
                for feat in feats:
                    if len(feat) <= 15:
                        embeds.append(embed.cpu().numpy())
                    else:
                        feat = feat.unsqueeze(0).to(device)
                        embed = generator(feat)
                        embeds.append(embed.cpu().numpy())
                embeds = np.vstack(embeds)
                embeds = normalize(embeds, axis=1)

                # Compute similarity matrix
                sim_matrix = pairwise_distances(embeds, metric='cosine')
                cluster_labels = agg_clustering_oracle(sim_matrix, num_spkrs)
                # TODO: Consider overlapped dataset, prototype for now

                # Write to rttm
                hyp_rttm_lines = make_rttm_lines(segcols, cluster_labels)


                lines_to_file(ref_rttm_lines, ref_rec_rttm)
                lines_to_file(hyp_rttm_lines, hyp_rec_rttm)

                # Eval based on recording level rttm
                der = score_der(hyp=hyp_rec_rttm, ref=ref_rec_rttm,
                                outfile='/tmp/{}.derlog'.format(rec), collar=0.25)
                print('({}) DER for {}: {}'.format(args.cfg, rec, der))

                rpkl[rec] = der
                pickle.dump(rpkl, open(results_pkl, 'wb'))

                all_ref_rttms.append(ref_rec_rttm)
                all_hyp_rttms.append(hyp_rec_rttm)

        final_hyp_rttm = os.path.join(args.model_dir, 'final_{}_hyp.rttm'.format(args.checkpoint))
        final_ref_rttm = os.path.join(args.model_dir, 'final_{}_ref.rttm'.format(args.checkpoint))

        os.system('cat {} > {}'.format(' '.join(all_ref_rttms), final_ref_rttm))
        os.system('cat {} > {}'.format(' '.join(all_hyp_rttms), final_hyp_rttm))

        time.sleep(4)

        full_der = score_der(hyp=final_hyp_rttm, ref=final_ref_rttm,
                             outfile=os.path.join(args.model_dir, 'final_{}_hyp.derlog'.format(args.checkpoint)), collar=0.25)
        print('({}) Full test DER: {}'.format(args.cfg, full_der))

        rpkl['full_der'] = full_der
        pickle.dump(rpkl, open(results_pkl, 'wb'))

        uem_file = os.path.join(TEST_DATA_PATH, 'uem')
        der_uem = score_der_uem(hyp=os.path.join(args.model_dir, 'final_{}_hyp.rttm'.format(args.checkpoint)),
                                ref=os.path.join(args.model_dir, 'final_{}_ref.rttm'.format(args.checkpoint)),
                                outfile=os.path.join(args.model_dir, 'final_hyp_uem.derlog'),
                                uem=uem_file,
                                collar=0.25)
        rpkl['full_der_uem'] = der_uem
        print('({}) Full test DER (uem): {}'.format(args.cfg, rpkl['full_der_uem']))
        pickle.dump(rpkl, open(results_pkl, 'wb'))


if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    uvloop.install()
    rpkl_path = os.path.join(args.model_dir, 'results.p')
    if not os.path.isfile(rpkl_path):
        print('No results.p found')
    else:
        device = torch.device('cuda')

        TEST_DATA_PATH = args.diar_data

        generator = setup()
        extract_and_diarize(generator, TEST_DATA_PATH, device)
