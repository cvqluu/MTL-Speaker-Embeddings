import argparse
import glob
import json
import os
import pickle
import random
import shutil
import sys
from collections import OrderedDict
from itertools import combinations

import numpy as np
from scipy.special import comb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prep the data for verification, diarization and feature extraction')
    parser.add_argument('--base-outfolder', type=str, help='Location of the base outfolder')
    parser.add_argument('--train-proportion', type=float, default=0.8, help='Train proportion (default: 0.8)')
    parser.add_argument('--pos-per-spk', type=int, default=15, help='Positive trials per speaker')
    args = parser.parse_args()
    return args


def load_n_col(file):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(' '))
    columns = list(zip(*data))
    columns = [list(i) for i in columns]
    return columns


def write_lines(lines, file):
    with open(file, 'w+') as fp:
        for line in lines:
            fp.write(line)


def fix_data_dir(data_dir):
    """
    Files: real_utt2spk, real_spk2utt, utt2age, wav.scp, utt2spk, spk2utt, segments
    Cleans and fixes all the right files to agree with kaldi data processing
    """
    backup_data_dir = os.path.join(data_dir, '.mybackup')
    os.makedirs(backup_data_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(data_dir, '*'))
    files = [f for f in files if os.path.isfile(f)]
    _ = [shutil.copy(f, backup_data_dir) for f in files]

    utt2spk_dict = OrderedDict({k: v for k, v in zip(*load_n_col(os.path.join(data_dir, 'real_utt2spk')))})
    utt2fpath_dict = OrderedDict({k: v for k, v in zip(*load_n_col(os.path.join(data_dir, 'feats.scp')))})

    utt2age = os.path.isfile(os.path.join(data_dir, 'utt2age'))
    if utt2age:
        utt2age_dict = OrderedDict({k: v for k, v in zip(*load_n_col(os.path.join(data_dir, 'utt2age')))})

    complete_utts = set(utt2fpath_dict.keys()).intersection(set(utt2spk_dict.keys()))
    complete_utts = sorted(list(complete_utts))

    print('Reducing real utt2spk ({}) to {} utts...'.format(len(utt2spk_dict),
                                                            len(complete_utts)))

    blank_utts = os.path.join(data_dir, 'utts')
    with open(blank_utts, 'w+') as fp:
        for u in complete_utts:
            fp.write('{}\n'.format(u))

    os.system('./filter_scp.pl {} {} > {}'.format(blank_utts, os.path.join(data_dir, 'real_utt2spk'),
                                                  os.path.join(data_dir, 'utt2spk')))

    os.rename(os.path.join(data_dir, 'segments'), os.path.join(data_dir, 'segments_old'))
    os.system('./filter_scp.pl {} {} > {}'.format(blank_utts, os.path.join(data_dir, 'segments_old'),
                                                  os.path.join(data_dir, 'segments')))

    if utt2age:
        os.rename(os.path.join(data_dir, 'utt2age'), os.path.join(data_dir, 'utt2age_old'))
        os.system('./filter_scp.pl {} {} > {}'.format(blank_utts, os.path.join(data_dir, 'utt2age'),
                                                  os.path.join(data_dir, 'utt2age')))


    set_spks = sorted(list(set(utt2spk_dict.values())))
    spk2utt_dict = OrderedDict({k: [] for k in set_spks})
    for u in complete_utts:
        spk2utt_dict[utt2spk_dict[u]].append(u)

    with open(os.path.join(data_dir, 'spk2utt'), 'w+') as fp:
        for s in spk2utt_dict:
            if spk2utt_dict[s]:
                line = '{} {}\n'.format(s, ' '.join(spk2utt_dict[s]))
                fp.write(line)


def load_one_tomany(file):
    one = []
    many = []
    with open(file) as fp:
        for line in fp:
            line = line.strip().split(' ', 1)
            one.append(line[0])
            m = line[1].split(' ')
            many.append(m)
    return one, many


def utt2spk_to_spk2utt(utt2spk_path, outfile=None):
    utts = []
    spks = []
    with open(utt2spk_path) as fp:
        for line in fp:
            splitup = line.strip().split(' ')
            assert len(splitup) == 2, 'Got more or less columns that was expected: (Got {}, expected 2)'.format(
                len(splitup))
            utts.append(splitup[0])
            spks.append(splitup[1])
    set_spks = sorted(list(set(spks)))
    spk2utt_dict = OrderedDict({k: [] for k in set_spks})
    for u, s in zip(utts, spks):
        spk2utt_dict[s].append(u)

    if outfile:
        with open(outfile, 'w+') as fp:
            for spk in spk2utt_dict:
                line = '{} {}\n'.format(spk, ' '.join(spk2utt_dict[spk]))
                fp.write(line)
    return spk2utt_dict


def split_recordings(data_dir, train_proportion=0.8):
    """
    Split the recordings based on train proportion

    returns train_recordings, test_recordings
    """
    np.random.seed(1234)
    random.seed(1234)
    segments_path = os.path.join(data_dir, 'segments')
    assert os.path.isfile(segments_path), "Couldn't find {}".format(segments_path)

    utts, urecs, _, _ = load_n_col(segments_path)
    
    setrecs = sorted(list(set(urecs)))
    num_train_recs = int(np.floor(train_proportion * len(setrecs)))

    train_recs = np.random.choice(setrecs, size=num_train_recs, replace=False)
    test_recs = [r for r in setrecs if r not in train_recs]
    return train_recs, test_recs


def split_data_dir(data_dir, train_recs, test_recs):
    """
    Split recordings and files into train and test subfolders based on train_recs, test_recs

    Filters feats, utt2spk, spk2utt, segments, (rttm, utt2age)
    """
    segments_path = os.path.join(data_dir, 'segments')
    assert os.path.isfile(segments_path), "Couldn't find {}".format(segments_path)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    utts, urecs, _, _ = load_n_col(segments_path)
    utt2rec_dict = OrderedDict({k:v for k,v in zip(utts, urecs)})

    train_utts = [u for u in utts if utt2rec_dict[u] in train_recs]
    test_utts = [u for u in utts if utt2rec_dict[u] in test_recs]

    tr_u = os.path.join(train_dir, 'utts')
    with open(tr_u, 'w+') as fp:
        for u in train_utts:
            fp.write('{}\n'.format(u))

    te_u = os.path.join(test_dir, 'utts')
    with open(te_u, 'w+') as fp:
        for u in test_utts:
            fp.write('{}\n'.format(u))

    os.system('./filter_scp.pl {} {} > {}'.format(tr_u, os.path.join(data_dir, 'utt2spk'),
                                                  os.path.join(train_dir, 'utt2spk')))
    os.system('./filter_scp.pl {} {} > {}'.format(te_u, os.path.join(data_dir, 'utt2spk'),
                                                  os.path.join(test_dir, 'utt2spk')))
    
    os.system('./filter_scp.pl {} {} > {}'.format(tr_u, os.path.join(data_dir, 'feats.scp'),
                                                  os.path.join(train_dir, 'feats.scp')))
    os.system('./filter_scp.pl {} {} > {}'.format(te_u, os.path.join(data_dir, 'feats.scp'),
                                                  os.path.join(test_dir, 'feats.scp')))

    os.system('./filter_scp.pl {} {} > {}'.format(tr_u, os.path.join(data_dir, 'segments'),
                                                  os.path.join(train_dir, 'segments')))
    os.system('./filter_scp.pl {} {} > {}'.format(te_u, os.path.join(data_dir, 'segments'),
                                                  os.path.join(test_dir, 'segments')))

    utt2spk_to_spk2utt(os.path.join(train_dir, 'utt2spk'), outfile=os.path.join(train_dir, 'spk2utt'))
    utt2spk_to_spk2utt(os.path.join(test_dir, 'utt2spk'), outfile=os.path.join(test_dir, 'spk2utt'))              

    if os.path.isfile(os.path.join(data_dir, 'utt2age')):
        os.system('./filter_scp.pl {} {} > {}'.format(tr_u, os.path.join(data_dir, 'utt2age'),
                                                  os.path.join(train_dir, 'utt2age')))
        os.system('./filter_scp.pl {} {} > {}'.format(te_u, os.path.join(data_dir, 'utt2age'),
                                                  os.path.join(test_dir, 'utt2age')))

    rttm_path = os.path.join(data_dir, 'ref.rttm')
    if os.path.isfile(rttm_path):
        with open(rttm_path) as fp:
            rttm_lines = [line for line in fp]

        rttm_recs = [line.split()[1].strip() for line in rttm_lines]

        tr_rttmlines = [l for l, r in zip(rttm_lines, rttm_recs) if r in train_recs]
        te_rttmlines = [l for l, r in zip(rttm_lines, rttm_recs) if r in test_recs]

        write_lines(tr_rttmlines, os.path.join(train_dir, 'ref.rttm'))
        write_lines(te_rttmlines, os.path.join(test_dir, 'ref.rttm'))


def nonoverlapped_utts(veri_data_dir):
    '''
    Retrieve only the utterances that belong to speakers not seen in the training set
    '''
    train_dir = os.path.join(veri_data_dir, 'train')
    test_dir = os.path.join(veri_data_dir, 'test')

    train_utts, train_spkrs = load_n_col(os.path.join(train_dir, 'utt2spk'))
    test_utts, test_spkrs = load_n_col(os.path.join(test_dir, 'utt2spk'))

    set_tr_spkrs = set(train_spkrs)
    set_te_spkrs = set(test_spkrs)

    non_overlapped_speakers = set_te_spkrs - set_tr_spkrs
    assert len(non_overlapped_speakers) >= 10, "Something has gone wrong if less than 10 speakers are left"

    valid_utts, valid_spkrs = [], []
    for u, s in zip(test_utts, test_spkrs):
        if s in non_overlapped_speakers:
            valid_utts.append(u)
            valid_spkrs.append(s)
    
    return valid_utts, valid_spkrs


def generate_veri_pairs(utts, spkrs, pos_per_spk=15):
    # Randomly creates pairs for a verification list
    # pos_per_spk determines the number of same-speaker pairs, which is always paired with an equal number of negatives
    np.random.seed(1234)
    random.seed(1234)
    setspkrs = sorted(list(set(spkrs)))
    spk2utt_dict = OrderedDict({s: [] for s in setspkrs})
    for utt, s in zip(utts, spkrs):
        spk2utt_dict[s].append(utt)

    u0 = []
    u1 = []
    labs = []

    for s in setspkrs:
        random.shuffle(spk2utt_dict[s])

    for s in setspkrs:
        positives = [spk2utt_dict[s].pop() for _ in range(pos_per_spk)]
        num_neg_trials = int(comb(pos_per_spk, 2))
        negatives = [spk2utt_dict[np.random.choice(list(set(setspkrs) - set([s])))].pop() for _ in
                     range(num_neg_trials)]
        for a, b in combinations(positives, 2):
            u0.append(a)
            u1.append(b)
            labs.append(1)
        for a, b in zip(np.random.choice(positives, size=num_neg_trials, replace=True), negatives):
            u0.append(a)
            u1.append(b)
            labs.append(0)

    return u0, u1, labs


if __name__ == "__main__":
    args = parse_args()

    veri_data_dir = os.path.join(args.base_outfolder, 'veri_data_nosil')
    assert os.path.isdir(veri_data_dir), "Couldn't find {}".format(veri_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'veri_data/segments'), veri_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'veri_data/utt2age'), veri_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'veri_data/real_utt2spk'), veri_data_dir)

    print('Fixing data dir: {}'.format(veri_data_dir))
    fix_data_dir(veri_data_dir)
    train_recs, test_recs = split_recordings(veri_data_dir, train_proportion=args.train_proportion)

    print('Making recording split...')
    with open(os.path.join(args.base_outfolder, 'recording_split.json'), 'w+', encoding='utf-8') as fp:
        recdict = {'train': train_recs, 'test': test_recs}
        json.dump(recdict, fp)
    
    print('Splitting verification data...')
    split_data_dir(veri_data_dir, train_recs, test_recs)

    print('Making verification pairs for test portion of verification data...')
    valid_utts, valid_spkrs = nonoverlapped_utts(veri_data_dir)
    u0, u1, labs = generate_veri_pairs(valid_utts, valid_spkrs, pos_per_spk=args.pos_per_spk)
    veri_lines = ['{} {} {}\n'.format(l, a, b) for l, a, b in zip(labs, u0, u1)]
    write_lines(veri_lines, os.path.join(veri_data_dir, 'test/veri_pairs'))

    print('Now fixing diarization data...')
    diar_data_dir = os.path.join(args.base_outfolder, 'diar_data_nosil')
    assert os.path.isdir(diar_data_dir), "Couldn't find {}".format(diar_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'diar_data/real_utt2spk'), veri_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'veri_data/segments'), veri_data_dir)
    shutil.copy(os.path.join(args.base_outfolder, 'diar_data/ref.rttm'), veri_data_dir)

    fix_data_dir(diar_data_dir)

    print('Splitting diarization data...')
    split_data_dir(diar_data_dir, train_recs, test_recs)


    print('Done!!')

    

