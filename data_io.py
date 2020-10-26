import asyncio
import itertools
import os
import pickle
import random
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from kaldi_io import read_mat, read_vec_flt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

from tqdm import tqdm

class MissingClassMapError(Exception):
    pass

def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(' '))
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns

def odict_from_2_col(file, numpy=False):
    col0, col1 = load_n_col(file, numpy=numpy)
    return OrderedDict({c0: c1 for c0, c1 in zip(col0, col1)})

def load_one_tomany(file, numpy=False):
    one = []
    many = []
    with open(file) as fp:
        for line in fp:
            line = line.strip().split(' ', 1)
            one.append(line[0])
            m = line[1].split(' ')
            many.append(np.array(m) if numpy else m)
    if numpy:
        one = np.array(one)
    return one, many

def train_transform(feats, seqlen):
    leeway = feats.shape[0] - seqlen
    startslice = np.random.randint(0, int(leeway)) if leeway > 0  else 0
    feats = feats[startslice:startslice+seqlen] if leeway > 0 else np.pad(feats, [(0,-leeway), (0,0)], 'constant')
    return torch.FloatTensor(feats)

async def get_item_train(instructions):
    fpath = instructions[0]
    seqlen = instructions[1]
    raw_feats = read_mat(fpath)
    feats = train_transform(raw_feats, seqlen)
    return feats

async def get_item_test(filepath):
    raw_feats = read_mat(filepath)
    return torch.FloatTensor(raw_feats)

def async_map(coroutine_func, iterable):
    loop = asyncio.get_event_loop()
    future = asyncio.gather(*(coroutine_func(param) for param in iterable))
    return loop.run_until_complete(future)


class SpeakerDataset(Dataset):

    def __init__(self, data_base_path,
                    real_speaker_labels=True,
                    asynchr=True, num_workers=3,
                    test_mode=False, class_enc_dict=None,
                    **kwargs):
        self.data_base_path = data_base_path
        self.num_workers = num_workers
        self.test_mode = test_mode
        self.real_speaker_labels = real_speaker_labels
        # self.label_types = label_types
        if self.test_mode:
            self.label_types = []
        else:
            self.label_types = ['speaker'] if self.real_speaker_labels else []

        if os.path.isfile(os.path.join(data_base_path, 'spk2nat')):
            self.label_types.append('nationality')

        if os.path.isfile(os.path.join(data_base_path, 'spk2gender')):
            self.label_types.append('gender')

        if os.path.isfile(os.path.join(data_base_path, 'utt2age')):
            self.label_types.append('age_regression')
            self.label_types.append('age')
        
        if os.path.isfile(os.path.join(data_base_path, 'utt2rec')):
            self.label_types.append('rec')

        if self.test_mode and self.label_types:
            assert class_enc_dict, 'Class mapping must be passed to test mode dataset'
            self.class_enc_dict = class_enc_dict

        utt2spk_path = os.path.join(data_base_path, 'utt2spk')
        spk2utt_path = os.path.join(data_base_path, 'spk2utt')
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')

        assert os.path.isfile(utt2spk_path)
        assert os.path.isfile(feats_scp_path)
        assert os.path.isfile(spk2utt_path)

        verilist_path = os.path.join(data_base_path, 'veri_pairs')

        if self.test_mode:
            if os.path.isfile(verilist_path):
                self.veri_labs, self.veri_0, self.veri_1 = load_n_col(verilist_path, numpy=True)
                self.veri_labs = self.veri_labs.astype(int)
                self.veripairs = True
            else:
                self.veripairs = False

        self.utts, self.uspkrs = load_n_col(utt2spk_path)
        self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.label_enc = LabelEncoder()

        self.original_spkrs, self.spkutts = load_one_tomany(spk2utt_path)
        self.spkrs = self.label_enc.fit_transform(self.original_spkrs)
        self.spk_utt_dict = OrderedDict({k:v for k,v in zip(self.spkrs, self.spkutts)})

        self.uspkrs = self.label_enc.transform(self.uspkrs)
        self.utt_spkr_dict = OrderedDict({k:v for k,v in zip(self.utts, self.uspkrs)})

        self.utt_list = list(self.utt_fpath_dict.keys())
        self.first_batch = True

        self.num_classes = {'speaker': len(self.label_enc.classes_)} if self.real_speaker_labels else {}
        self.asynchr = asynchr

        if 'nationality' in self.label_types:
            self.natspkrs, self.nats = load_n_col(os.path.join(data_base_path, 'spk2nat'))
            self.nats = [n.lower().strip() for n in self.nats]
            self.natspkrs = self.label_enc.transform(self.natspkrs)
            self.nat_label_enc = LabelEncoder()

            if not self.test_mode:
                self.nats = self.nat_label_enc.fit_transform(self.nats)
            else:
                self.nat_label_enc = self.class_enc_dict['nationality']
                self.nats = self.nat_label_enc.transform(self.nats)

            self.spk_nat_dict = OrderedDict({k:v for k,v in zip(self.natspkrs, self.nats)})
            self.num_classes['nationality'] = len(self.nat_label_enc.classes_)

        if 'gender' in self.label_types:
            self.genspkrs, self.genders = load_n_col(os.path.join(data_base_path, 'spk2gender'))
            self.genspkrs = self.label_enc.transform(self.genspkrs)
            self.gen_label_enc = LabelEncoder()

            if not self.test_mode:
                self.genders = self.gen_label_enc.fit_transform(self.genders)
            else:
                self.gen_label_enc = self.class_enc_dict['gender']
                self.genders = self.gen_label_enc.transform(self.genders)

            self.spk_gen_dict = OrderedDict({k:v for k,v in zip(self.genspkrs, self.genders)})
            self.num_classes['gender'] = len(self.gen_label_enc.classes_)

        if 'age' in self.label_types:
            # self.genspkrs, self.genders = load_n_col(os.path.join(data_base_path, 'spk2gender'))
            self.num_age_bins = kwargs['num_age_bins'] if 'num_age_bins' in kwargs else 10
            self.ageutts, self.ages = load_n_col(os.path.join(data_base_path, 'utt2age'))
            self.ages = np.array(self.ages).astype(np.float)
            self.age_label_enc = KBinsDiscretizer(n_bins=self.num_age_bins, encode='ordinal', strategy='uniform')

            if not self.test_mode:
                self.age_classes = self.age_label_enc.fit_transform(np.array(self.ages).reshape(-1, 1)).flatten()
            else:
                self.age_label_enc = self.class_enc_dict['age']
                self.age_classes = self.age_label_enc.transform(np.array(self.ages).reshape(-1, 1)).flatten()

            self.utt_age_class_dict = OrderedDict({k:v for k,v in zip(self.ageutts, self.age_classes)})
            self.num_classes['age'] = self.num_age_bins

        if 'age_regression' in self.label_types:
            # self.genspkrs, self.genders = load_n_col(os.path.join(data_base_path, 'spk2gender'))
            self.ageutts, self.ages = load_n_col(os.path.join(data_base_path, 'utt2age'))
            self.ages = np.array(self.ages).astype(np.float)

            self.utt_age_dict = OrderedDict({k:v for k,v in zip(self.ageutts, self.ages)})
            self.num_classes['age_regression'] = 1

        if 'rec' in self.label_types:
            self.recutts, self.recs = load_n_col(os.path.join(data_base_path, 'utt2rec'))
            self.recs = np.array(self.recs)
            self.rec_label_enc = LabelEncoder()

            if not self.test_mode:
                self.recs = self.rec_label_enc.fit_transform(self.recs)
            else:
                self.rec_label_enc = self.class_enc_dict['rec']
                self.recs = self.rec_label_enc.transform(self.recs)
            
            self.utt_rec_dict = OrderedDict({k:v for k,v in zip(self.recutts, self.recs)})
            self.num_classes['rec'] = len(self.rec_label_enc.classes_)

        self.class_enc_dict = self.get_class_encs()

    def __len__(self):
        return len(self.utt_list)


    def get_class_encs(self):
        class_enc_dict = {}
        if 'speaker' in self.label_types:
            class_enc_dict['speaker'] = self.label_enc
        if 'age' in self.label_types:
            class_enc_dict['age'] = self.age_label_enc
        if 'age_regression' in self.label_types:
            class_enc_dict['age_regression'] = None
        if 'nationality' in self.label_types:
            class_enc_dict['nationality'] = self.nat_label_enc
        if 'gender' in self.label_types:
            class_enc_dict['gender'] = self.gen_label_enc
        if 'rec' in self.label_types:
            class_enc_dict['rec'] = self.rec_label_enc
        self.class_enc_dict = class_enc_dict
        return class_enc_dict


    @staticmethod
    def get_item(instructions):
        fpath = instructions[0]
        seqlen = instructions[1]
        feats = read_mat(fpath)
        feats = train_transform(feats, seqlen)
        return feats


    def get_item_test(self, idx):
        utt = self.utt_list[idx]
        fpath = self.utt_fpath_dict[utt]
        feats = read_mat(fpath)
        feats = torch.FloatTensor(feats)

        label_dict = {}
        speaker = self.utt_spkr_dict[utt]

        if 'speaker' in self.label_types:
            label_dict['speaker'] = torch.LongTensor([speaker])
        if 'gender' in self.label_types:
            label_dict['gender'] = torch.LongTensor([self.spk_gen_dict[speaker]])
        if 'nationality' in self.label_types:
            label_dict['nationality'] = torch.LongTensor([self.spk_nat_dict[speaker]])
        if 'age' in self.label_types:
            label_dict['age'] = torch.LongTensor([self.utt_age_class_dict[utt]])
        if 'age_regression' in self.label_types:
            label_dict['age_regression'] = torch.FloatTensor([self.utt_age_dict[utt]])

        return feats, label_dict


    def get_test_items(self):
        utts = self.utt_list
        fpaths = [self.utt_fpath_dict[utt] for utt in utts]
        feats = async_map(get_item_test, fpaths)

        label_dict = {}
        spkrs = [self.utt_spkr_dict[utt] for utt in utts]

        if 'speaker' in self.label_types:
            label_dict['speaker'] = np.array(spkrs)
        if 'nationality' in self.label_types:
            label_dict['nationality'] = np.array([self.spk_nat_dict[s] for s in spkrs])
        if 'gender' in self.label_types:
            label_dict['gender'] = np.array([self.spk_gen_dict[s] for s in spkrs])
        if 'age' in self.label_types:
            label_dict['age'] = np.array([self.utt_age_class_dict[utt] for utt in utts])
        if 'age_regression' in self.label_types:
            label_dict['age_regression'] = np.array([self.utt_age_dict[utt] for utt in utts])

        return feats, label_dict, utts


    def get_batches(self, batch_size=256, max_seq_len=400, sp_tensor=True):
        """
        Main data iterator, specify batch_size and max_seq_len
        sp_tensor determines whether speaker labels are returned as Tensor object or not
        """
        # with Parallel(n_jobs=self.num_workers) as parallel:
        self.idpool = self.spkrs.copy()
        assert batch_size < len(self.idpool) #Metric learning assumption large num classes
        lens = [max_seq_len for _ in range(batch_size)]
        while True:
            if len(self.idpool) <= batch_size:
                batch_ids = np.array(self.idpool)
                self.idpool = self.spkrs.copy()
                rem_ids = np.random.choice(self.idpool, size=batch_size-len(batch_ids), replace=False)
                batch_ids = np.concatenate([batch_ids, rem_ids])
                self.idpool = list(set(self.idpool) - set(rem_ids))
            else:
                batch_ids = np.random.choice(self.idpool, size=batch_size, replace=False)
                self.idpool = list(set(self.idpool) - set(batch_ids))

            batch_fpaths = []
            batch_utts = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_utts.append(utt)
                batch_fpaths.append(self.utt_fpath_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
            # batch_feats = parallel(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))

            label_dict = {}
            if 'speaker' in self.label_types:
                label_dict['speaker'] = torch.LongTensor(batch_ids) if sp_tensor else batch_ids
            if 'nationality' in self.label_types:
                label_dict['nationality'] = torch.LongTensor([self.spk_nat_dict[s] for s in batch_ids])
            if 'gender' in self.label_types:
                label_dict['gender'] = torch.LongTensor([self.spk_gen_dict[s] for s in batch_ids])
            if 'age' in self.label_types:
                label_dict['age'] = torch.LongTensor([self.utt_age_class_dict[u] for u in batch_utts])
            if 'age_regression' in self.label_types:
                label_dict['age_regression'] = torch.FloatTensor([self.utt_age_dict[u] for u in batch_utts])
            if 'rec' in self.label_types:
                label_dict['rec'] = torch.LongTensor([self.utt_rec_dict[u] for u in batch_utts])
                
            yield torch.stack(batch_feats), label_dict



class SpeakerDatasetMultiNC(Dataset):

    def __init__(self, datasets, dslabels, sample_weights=None):
        """
        Collator of different SpeakerDataset classes, No combination of classes
        """
        assert len(datasets) >= 1, '1 or more datasets must be supplied'
        assert len(datasets) == len(dslabels), 'Datasets need to be labelled'
        self.datasets = datasets
        self.dslabels = dslabels

        self.num_class_dicts = []
        self.utt_list = []
        self.label_types = []
        for ds in self.datasets:
            assert isinstance(ds, SpeakerDataset), "Each object in datasets must be SpeakerDataset object"
            self.num_class_dicts.append(ds.num_classes)
            self.utt_list += ds.utt_list
            self.label_types += ds.label_types

        # assert len(self.label_types) - len(set(self.label_types)) <= 1, "Only speaker allowed to be unique, datasets must have different attributes"
        # self.label_types = list(set(self.label_types))

        self.num_datasets = len(datasets)
        self.sample_weights = np.ones(self.num_datasets)/self.num_datasets if not sample_weights else sample_weights
        self.sample_weights = self.sample_weights/np.sum(self.sample_weights)

        self.num_classes = {}
        for n, dsl in zip(self.num_class_dicts, self.dslabels):
            for k in n:
                label_name = '{}_{}'.format(dsl, k)
                self.num_classes[label_name] = n[k]



    def __len__(self):
        return len(self.utt_list)

    def get_batches(self, batch_size=256, max_seq_len=400):
        """
        Data iterator that collates all of self.datasets
        Yields the following:
          - collated input features
          - list of label dicts
          - index markers of where to slice batch for different datasets
        """
        ds_batch_sizes = np.floor(self.sample_weights * batch_size).astype(int)

        while np.sum(ds_batch_sizes) < batch_size:
            for i in range(self.num_datasets):
                if np.sum(ds_batch_sizes) < batch_size:
                    ds_batch_sizes[i] += 1

        assert np.sum(ds_batch_sizes) == batch_size, "Batch size doesn't match"

        data_providers = [ds.get_batches(batch_size=mini_bs, max_seq_len=max_seq_len, sp_tensor=False) for ds, mini_bs in zip(self.datasets, ds_batch_sizes)]
        index_markers = []
        start_idx = 0
        for mini_bs in ds_batch_sizes:
            index_markers.append((start_idx, start_idx + mini_bs))
            start_idx += mini_bs

        while True:
            batch_feats = []
            label_dicts = []

            for it, ds in zip(data_providers, self.datasets):
                bf, ld = next(it)
                batch_feats.append(bf)
                label_dicts.append(ld)

            yield torch.cat(batch_feats, dim=0), label_dicts, index_markers


class SpeakerDatasetMulti(Dataset):

    def __init__(self, datasets, sample_weights=None):
        """
        Collator of different SpeakerDataset classes
        """
        assert len(datasets) >= 1, '1 or more datasets must be supplied'
        self.datasets = datasets

        self.num_class_dicts = []
        self.utt_list = []
        self.label_types = []
        for ds in self.datasets:
            assert isinstance(ds, SpeakerDataset), "Each object in datasets must be SpeakerDataset object"
            self.num_class_dicts.append(ds.num_classes)
            self.utt_list += ds.utt_list
            self.label_types += ds.label_types

        assert len(self.label_types) - len(set(self.label_types)) <= 1, "Only speaker allowed to be unique, datasets must have different attributes"
        self.label_types = list(set(self.label_types))

        self.num_datasets = len(datasets)
        self.sample_weights = np.ones(self.num_datasets)/self.num_datasets if not sample_weights else sample_weights
        self.sample_weights = self.sample_weights/np.sum(self.sample_weights)

        self.num_classes = {'speaker': 0}
        for n in self.num_class_dicts:
            for k in n:
                if k != 'speaker':
                    self.num_classes[k] = n[k]
        

        self.label_enc = LabelEncoder()

        self.class_enc_dict = {}
        self.all_speaker_classes = []
        
        for ds in self.datasets:
            ds_cls_enc = ds.get_class_encs()
            for l in ds_cls_enc:
                if l ==  'speaker':
                    self.all_speaker_classes.append(ds_cls_enc['speaker'].classes_)
                else:
                    self.class_enc_dict[l] = ds_cls_enc[l]
        
        self.all_speaker_classes = np.concatenate(self.all_speaker_classes)
        self.label_enc.fit_transform(self.all_speaker_classes)
        self.class_enc_dict['speaker'] = self.label_enc
        self.num_classes['speaker'] = len(self.label_enc.classes_)


    def __len__(self):
        return len(self.utt_list)

    def get_batches(self, batch_size=256, max_seq_len=400):
        """
        Data iterator that collates all of self.datasets
        Yields the following:
          - collated input features
          - list of label dicts
          - index markers of where to slice batch for different datasets
        """
        ds_batch_sizes = np.floor(self.sample_weights * batch_size).astype(int)

        while np.sum(ds_batch_sizes) < batch_size:
            for i in range(self.num_datasets):
                if np.sum(ds_batch_sizes) < batch_size:
                    ds_batch_sizes[i] += 1

        assert np.sum(ds_batch_sizes) == batch_size, "Batch size doesn't match"

        data_providers = [ds.get_batches(batch_size=mini_bs, max_seq_len=max_seq_len, sp_tensor=False) for ds, mini_bs in zip(self.datasets, ds_batch_sizes)]
        index_markers = []
        start_idx = 0
        for mini_bs in ds_batch_sizes:
            index_markers.append((start_idx, start_idx + mini_bs))
            start_idx += mini_bs

        while True:
            batch_feats = []
            label_dicts = []

            for it, ds in zip(data_providers, self.datasets):
                bf, ld = next(it)
                if 'speaker' in ld:
                    orig_sp_labels = ds.class_enc_dict['speaker'].inverse_transform(ld['speaker'])
                    ld['speaker'] = torch.LongTensor(self.class_enc_dict['speaker'].transform(orig_sp_labels))

                batch_feats.append(bf)
                label_dicts.append(ld)

            yield torch.cat(batch_feats, dim=0), label_dicts, index_markers


class SpeakerTestDataset(Dataset):

    def __init__(self, data_base_path, asynchr=True):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        verilist_path = os.path.join(data_base_path, 'veri_pairs')
        utt2spk_path = os.path.join(data_base_path, 'utt2spk')

        assert os.path.isfile(verilist_path)

        if os.path.isfile(feats_scp_path):
            self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.veri_labs, self.veri_0, self.veri_1 = load_n_col(verilist_path, numpy=True)
        self.utt2spk_dict = odict_from_2_col(utt2spk_path)
        self.enrol_utts = list(set(self.veri_0))
        self.veri_utts = sorted(list(set(np.concatenate([self.veri_0, self.veri_1]))))
        self.veri_labs = self.veri_labs.astype(int)

    def __len__(self):
        return len(self.veri_labs)

    def __getitem__(self, idx):
        utt = self.veri_utts[idx]
        fpath = self.utt_fpath_dict[utt]
        feats = torch.FloatTensor(read_mat(fpath))
        return feats, utt

    def get_test_items(self):
        utts = self.veri_utts
        fpaths = [self.utt_fpath_dict[utt] for utt in utts]
        feats = async_map(get_item_test, fpaths)
        # feats = [torch.FloatTensor(read_mat(fpath)) for fpath in fpaths]
        return feats, utts


class DiarizationDataset(Dataset):

    def __init__(self, data_base_path, asynchr=True):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        utt2spk_path = os.path.join(data_base_path, 'utt2spk')
        segments_path = os.path.join(data_base_path, 'segments')
        self.ref_rttm_path = os.path.join(data_base_path, 'ref.rttm')

        assert os.path.isfile(feats_scp_path)
        assert os.path.isfile(utt2spk_path)
        assert os.path.isfile(segments_path)
        assert os.path.isfile(self.ref_rttm_path)

        self.utt_fpath_dict = odict_from_2_col(feats_scp_path)
        self.utt2spk_dict = odict_from_2_col(utt2spk_path)
        
        self.segcols = load_n_col(segments_path, numpy=True)
        self.segutts = self.segcols[0]
        assert set(self.segutts) == set(self.utt2spk_dict.keys())
        self.segrecs = self.segcols[1]
        self.recs = sorted(set(self.segrecs))

        self.utt2rec_dict = OrderedDict({k:v for k,v in zip(self.segutts, self.segrecs)})
        self.rec2utt_dict = OrderedDict({k:[] for k in self.recs})
        for u in self.utt2rec_dict:
            self.rec2utt_dict[self.utt2rec_dict[u]].append(u)

        self.rttmcols = load_n_col(self.ref_rttm_path, numpy=True)
        self.rttmcols = self.remove_bad_rttm_rows(self.rttmcols)
        self.rttm_recs = self.rttmcols[1]
        assert len(self.recs) == len(set(self.rttm_recs))

    def __len__(self):
        return len(self.setrecs)

    @staticmethod
    def remove_bad_rttm_rows(rttmcols):
        assert len(rttmcols) == 10, 'expected 10 rttm columns'
        ends = np.array(rttmcols[4]).astype(float)
        good_rows = ends > 0.0
        original_len = len(rttmcols[0])
        get_lines = lambda a: a[good_rows]
        newcols = [get_lines(c) for c in rttmcols]
        final_len = len(newcols[0])
        print('Removed {} malformed/bad rows from rttm'.format(original_len - final_len))
        return newcols

    @staticmethod
    def sim_matrix_target(labels):
        le = LabelEncoder()
        dist = 1.0 - pairwise_distances(le.fit_transform(labels)[:,np.newaxis], metric='hamming')
        return dist

    @staticmethod
    def cols_to_lines(cols):
        lines = [' '.join(r) + '\n' for r in zip(*cols)]
        return lines

    @staticmethod
    def segment_entries_to_rttmlines(segcols):
        rttmline = 'SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n'
        recs = segcols[1]
        starts = segcols[2].astype(float)
        ends = segcols[3].astype(float)
        offsets = np.around(ends - starts, decimals=3)
        rttmline_p1 = 'SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> '
        rttmline_p2 = '{} <NA> <NA>\n'

        lines = []
        for r, s, o in zip(recs, starts, offsets):
            full_line = rttmline_p1.format(r, s, o) + rttmline_p2
            lines.append(full_line)
        return lines

    def rttm_lines_from_rec(self, rec):
        # Get reference rttm lines that are relevant to rec
        assert rec in self.recs
        reclines = self.rttm_recs == rec
        assert len(reclines) >= 1, '>= One line must be found'
        get_lines = lambda a: a[reclines]
        newcols = [get_lines(c) for c in self.rttmcols]
        return self.cols_to_lines(newcols)
    
    def segments_lines_from_rec(self, rec):
        # Get segments lines that are relevant to rec, formatted as rttm
        assert rec in self.recs
        reclines = self.segrecs == rec
        assert len(reclines) >= 1, '>= One line must be found'
        get_lines = lambda a: a[reclines]
        newcols = [get_lines(c) for c in self.segcols]
        return self.segment_entries_to_rttmlines(newcols), newcols[0]

    def segments_cols_from_rec(self, rec):
        # Get segments lines that are relevant to rec, formatted as rttm
        assert rec in self.recs
        reclines = self.segrecs == rec
        assert len(reclines) >= 1, '>= One line must be found'
        get_lines = lambda a: a[reclines]
        newcols = [get_lines(c) for c in self.segcols]
        return newcols

    def __getitem__(self, idx):
        rec = self.recs[idx]
        utts = self.rec2utt_dict[rec]
        spkrs = [self.utt2spk_dict[u] for u in utts]

        ref_rttm_lines = self.rttm_lines_from_rec(rec)
        # hyp_rttm_lines, segutts = self.segments_lines_from_rec(rec)
        # assert (segutts == utts).all()
        segcols = self.segments_cols_from_rec(rec)

        okay_feats = []
        okay_spkrs = []
        okay_idx = []

        fpaths = [self.utt_fpath_dict[utt] for utt in utts]
        for i, fpath in enumerate(fpaths):
            try:
                okay_feats.append(torch.FloatTensor(read_mat(fpath)))
                okay_spkrs.append(spkrs[i])
                okay_idx.append(i)
            except:
                print('Reading utterance {} failed'.format(utts[i]))
                continue

        okay_idx = np.array(okay_idx)
        get_lines = lambda a: a[okay_idx]
        newsegcols = [get_lines(c) for c in segcols]

        return okay_feats, okay_spkrs, ref_rttm_lines, newsegcols, rec


class SpeakerEvalDataset(Dataset):

    def __init__(self, data_base_path):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        model_enrollment_path = os.path.join(data_base_path, 'model_enrollment.txt')
        eval_veri_pairs_path = os.path.join(data_base_path, 'trials.txt')

        if os.path.isfile(feats_scp_path):
            self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.models, self.enr_utts = load_one_tomany(model_enrollment_path)
        if self.models[0] == 'model-id':
            self.models, self.enr_utts = self.models[1:], self.enr_utts[1:]
        assert len(self.models) == len(set(self.models))
        self.model_enr_utt_dict = OrderedDict({k:v for k,v in zip(self.models, self.enr_utts)})

        self.all_enrol_utts = list(itertools.chain.from_iterable(self.enr_utts))

        self.models_eval, self.eval_utts = load_n_col(eval_veri_pairs_path)
        if self.models_eval[0] == 'model-id':
            self.models_eval, self.eval_utts = self.models_eval[1:], self.eval_utts[1:]

        assert set(self.models_eval) == set(self.models)

        self.model_eval_utt_dict = OrderedDict({})
        for m, ev_utt in zip(self.models_eval, self.eval_utts):
            if m not in self.model_eval_utt_dict:
                self.model_eval_utt_dict[m] = []
            self.model_eval_utt_dict[m].append(ev_utt)

        self.models = list(self.model_eval_utt_dict.keys())

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        '''
        Returns enrolment utterances and eval utterances for a specific model
        '''
        model = self.models[idx]
        enrol_utts = self.model_enr_utt_dict[model]
        eval_utts = self.model_eval_utt_dict[model]
        enrol_fpaths = [self.utt_fpath_dict[u] for u in enrol_utts]
        eval_fpaths = [self.utt_fpath_dict[u] for u in eval_utts]

        enrol_feats = async_map(get_item_test, enrol_fpaths)
        eval_feats = async_map(get_item_test, eval_fpaths)

        return model, enrol_utts, enrol_feats, eval_utts, eval_feats

    def get_batches(self, utts):
        fpaths = [self.utt_fpath_dict[u] for u in utts]
        feats = async_map(get_item_test, fpaths)
        return feats

    def get_item_utts(self, idx):
        model = self.models[idx]
        enrol_utts = self.model_enr_utt_dict[model]
        eval_utts = self.model_eval_utt_dict[model]
        return model, enrol_utts, eval_utts


class SpeakerModelTestDataset(Dataset):

    def __init__(self, data_base_path):
        self.data_base_path = data_base_path
        feats_scp_path = os.path.join(data_base_path, 'feats.scp')
        model_utts_path = os.path.join(data_base_path, 'model_utts')
        model_ver_pairs_path = os.path.join(data_base_path, 'model_veri_pairs')

        if os.path.isfile(feats_scp_path):
            self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        # self.veri_labs, self.veri_0, self.veri_1 = load_n_col(verilist_path, numpy=True)
        self.models, self.m_utts = load_one_tomany(model_utts_path)
        self.all_enrol_utts = list(itertools.chain.from_iterable(self.m_utts))

        self.mvp = load_n_col(model_ver_pairs_path, numpy=True)
        if len(self.mvp) == 3:
            self.veri_labs, self.models_eval, self.eval_utts = self.mvp[0], self.mvp[1], self.mvp[2]
            self.veri_labs = np.array(self.veri_labs).astype(int)
        elif len(self.mvp) == 2:
            self.models_eval, self.eval_utts = self.mvp[0], self.mvp[1]
            self.veri_labs = None
        else:
            assert None, 'model_veri_pairs is in the wrong format'

        self.all_utts = self.all_enrol_utts + list(set(self.eval_utts))

    def __len__(self):
        return len(self.all_utts)

    def __getitem__(self, idx):
        utt = self.all_utts[idx]
        fpath = self.utt_fpath_dict[utt]
        feats = torch.FloatTensor(read_mat(fpath))
        return feats, utt

    def get_batches(self, utts):
        fpaths = [self.utt_fpath_dict[u] for u in utts]
        feats = async_map(get_item_test, fpaths)
        return feats

    def get_model_utts(self, idx):
        model = self.models[idx]
        utts = self.m_utts[idx]
        fpaths = [self.utt_fpath_dict[u] for u in utts]
        feats = async_map(get_item_test, fpaths)
        return feats

    def get_model_trials(self, idx):
        assert self.veri_labs is not None
        model = self.models[idx]
        indexes = self.models_eval == model
        labels = self.veri_labs[indexes]
        eval_utts = self.eval_utts[indexes]
        fpaths = [self.utt_fpath_dict[u] for u in eval_utts]
        feats = async_map(get_item_test, fpaths)
        return feats, labels



"""
Following is methods and classes for diarization models
"""


def read_xvec(file):
    return read_vec_flt(file)


async def async_read_xvec(path):
    return read_vec_flt(path)
    

def pairwise_cat_matrix(xvecs, labels):
    '''
    xvecs: (seq_len, d_xvec)
    labels: (seq_len)
    '''
    xvecs = np.array(xvecs)
    seq_len, d_xvec = xvecs.shape
    xproject = np.tile(xvecs, seq_len).reshape(seq_len, seq_len, d_xvec)
    yproject = np.swapaxes(xproject, 0, 1)
    matrix = np.concatenate([xproject, yproject], axis=-1)
    label_matrix = sim_matrix_target(labels)
    return np.array(matrix), label_matrix

def sim_matrix_target(labels):
    le = LabelEncoder()
    dist = 1.0 - pairwise_distances(le.fit_transform(labels)[:,np.newaxis], metric='hamming')
    return dist


def recombine_matrix(submatrices):
    dim = int(np.sqrt(len(submatrices)))
    rows = []
    for j in range(dim):
        start = j * dim
        row = np.concatenate(submatrices[start:start+dim], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def collate_sim_matrices(out_list, rec_ids):
    '''
    expect input list
    '''
    comb_matrices = []
    comb_ids = []
    matrix_buffer = []
    last_rec_id = rec_ids[0]
    for rid, vec in zip(rec_ids, out_list):
        if last_rec_id == rid:
            matrix_buffer.append(vec)
        else:
            if len(matrix_buffer) > 1:
                comb_matrices.append(recombine_matrix(matrix_buffer))
            else:
                comb_matrices.append(matrix_buffer[0])
            comb_ids.append(last_rec_id)
            matrix_buffer = [vec]
        last_rec_id = rid
    if len(matrix_buffer) > 1:
        comb_matrices.append(recombine_matrix(matrix_buffer))
    else:
        comb_matrices.append(matrix_buffer[0])
    comb_ids.append(last_rec_id)
    return comb_matrices, comb_ids


def batch_matrix(xvecpairs, labels, factor=2):
    baselen = len(labels)//factor
    split_batch = []
    split_batch_labs = []
    for j in range(factor):
        for i in range(factor):
            start_j = j * baselen
            end_j = (j+1) * baselen if j != factor - 1 else None
            start_i = i * baselen
            end_i = (i+1) * baselen if i != factor - 1 else None

            mini_pairs = xvecpairs[start_j:end_j, start_i:end_i, :]
            mini_labels = labels[start_j:end_j, start_i:end_i]

            split_batch.append(mini_pairs)
            split_batch_labs.append(mini_labels)
    return split_batch, split_batch_labs


def split_recs(rec_id, rlabs, rpaths, max_rec_len=1200):
    if len(rlabs) <= max_rec_len:
        return [[rec_id, rlabs, rpaths]]
    else:
        num_splits = int(np.ceil(len(rlabs)/max_rec_len))
        splits = []
        for i in range(num_splits):
            s_start = i * max_rec_len
            s_end = (i+1) * max_rec_len
            s_rec_id = rec_id + '({})'.format(i)
            s_rlabs = rlabs[s_start:s_end]
            s_rpaths = rpaths[s_start:s_end]
            splits.append([s_rec_id, s_rlabs, s_rpaths])
        return splits
    

def group_recs(utt2spk, segments, xvecscp, max_rec_len=1200):
    '''
    Groups utts with xvectors according to recording
    '''
    print('Loading utt2spk ...')
    utts, labels = load_n_col(utt2spk, numpy=True)
    uspkdict = {k:v for k,v in tqdm(zip(utts, labels), total=len(utts))}
    print('Loading xvector.scp ...')
    xutts, xpaths = load_n_col(xvecscp, numpy=True)
    xdict = {k:v for k,v in tqdm(zip(xutts, xpaths), total=len(xutts))}
    print('Loading segments ...')
    sutts, srecs, _, _ = load_n_col(segments, numpy=True)
    rec_ids = sorted(list(set(srecs)))

    assert len(xutts) >= len(sutts), 'Length mismatch: xvector.scp ({}) vs segments({})'.format(len(xutts), len(sutts))
    
    print('Sorting into recordings ...')
    rec_batches = []
    final_rec_ids = []
    for i in tqdm(rec_ids):
        rutts = sutts[srecs == i]
        rlabs = [uspkdict[u] for u in rutts]
        rpaths = [xdict[u] for u in rutts]
        s_batches = split_recs(i, rlabs, rpaths, max_rec_len=max_rec_len)
        for s in s_batches:
            rec_batches.append([s[1], s[2]])
            final_rec_ids.append(s[0])
    return final_rec_ids, rec_batches


class XVectorDataset:

    def __init__(self, data_path, xvector_scp=None, max_len=400, max_rec_len=1200, xvecbase_path=None, shuffle=True):
        self.data_path = data_path
        utt2spk = os.path.join(data_path, 'utt2spk')
        segments = os.path.join(data_path, 'segments')
        if not xvector_scp:
            xvecscp = os.path.join(data_path, 'xvector.scp')
        else:
            xvecscp = xvector_scp
    
        assert os.path.isfile(utt2spk)
        assert os.path.isfile(segments)
        assert os.path.isfile(xvecscp)
        
        self.ids, self.rec_batches = group_recs(utt2spk, segments, xvecscp, max_rec_len=max_rec_len)
        self.lengths = np.array([len(batch[0]) for batch in self.rec_batches])
        self.factors = np.ceil(self.lengths/max_len).astype(int)
        self.first_rec = np.argmax(self.lengths)
        self.max_len = max_len
        self.max_rec_len = max_rec_len
        self.shuffle = shuffle

    def __len__(self):
        return np.sum(self.factors**2)

    def get_batches(self):
        rec_order = np.arange(len(self.rec_batches))
        if self.shuffle:
            np.random.shuffle(rec_order)
            first_rec = np.argwhere(rec_order == self.first_rec).flatten()
            rec_order[0], rec_order[first_rec] = rec_order[first_rec], rec_order[0]

        for i in rec_order:
            rec_id = self.ids[i]
            labels, paths = self.rec_batches[i]
            xvecs = async_map(async_read_xvec, paths)
            pmatrix, plabels = pairwise_cat_matrix(xvecs, labels)
            if len(labels) <= self.max_len:
                yield pmatrix, plabels, rec_id
            else:
                factor = np.ceil(len(labels)/self.max_len).astype(int)
                batched_feats, batched_labels = batch_matrix(pmatrix, plabels, factor=factor)
                for feats, labels in zip(batched_feats, batched_labels):
                    yield feats, labels, rec_id


    def get_batches_seq(self):
        rec_order = np.arange(len(self.rec_batches))
        if self.shuffle:
            np.random.shuffle(rec_order)
            first_rec = np.argwhere(rec_order == self.first_rec).flatten()
            rec_order[0], rec_order[first_rec] = rec_order[first_rec], rec_order[0]
        for i in rec_order:
            rec_id = self.ids[i]
            labels, paths = self.rec_batches[i]
            xvecs = async_map(async_read_xvec, paths)
            # xvecs = np.array([read_xvec(file) for file in paths])
            pwise_labels = sim_matrix_target(labels)
            yield xvecs, pwise_labels, rec_id
