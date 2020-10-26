import argparse
import configparser
import os
import pickle
import sys
import h5py
import json

from collections import OrderedDict
from glob import glob
from math import floor, log10

import numpy as np

import kaldi_io
import torch
import uvloop
from data_io import SpeakerTestDataset, odict_from_2_col, SpeakerDataset
from kaldi_io import read_vec_flt
from kaldiio import ReadHelper
from models.extractors import ETDNN, FTDNN, XTDNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from utils import SpeakerRecognitionMetrics



def mtd(stuff, device):
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    else:
        return [mtd(s, device) for s in stuff]

def parse_args():
    parser = argparse.ArgumentParser(description='Test SV Model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--best', action='store_true', default=False, help='Use best model')
    parser.add_argument('--checkpoint', type=int, default=-1,  # which model to use, overidden by 'best'
                        help='Use model checkpoint, default -1 uses final model')
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    return args

def parse_config(args):
    assert os.path.isfile(args.cfg)
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_data = config['Datasets'].get('train')
    assert args.train_data
    args.test_data = config['Datasets'].get('test')

    args.model_type = config['Model'].get('model_type', fallback='XTDNN')
    assert args.model_type in ['XTDNN', 'ETDNN', 'FTDNN']

    args.classifier_heads = config['Model'].get('classifier_heads').split(',')
    assert len(args.classifier_heads) <= 3, 'Three options available'
    assert len(args.classifier_heads) == len(set(args.classifier_heads))
    for clf in args.classifier_heads:
        assert clf in ['speaker', 'nationality', 'gender', 'age', 'age_regression']

    args.classifier_types = config['Optim'].get('classifier_types').split(',')
    assert len(args.classifier_heads) == len(args.classifier_types)

    args.classifier_loss_weighting_type = config['Optim'].get('classifier_loss_weighting_type', fallback='none')
    assert args.classifier_loss_weighting_type in ['none', 'uncertainty_kendall', 'uncertainty_liebel', 'dwa']

    args.dwa_temperature = config['Optim'].getfloat('dwa_temperature', fallback=2.)

    args.classifier_loss_weights = np.array(json.loads(config['Optim'].get('classifier_loss_weights'))).astype(float)
    assert len(args.classifier_heads) == len(args.classifier_loss_weights)

    args.classifier_lr_mults = np.array(json.loads(config['Optim'].get('classifier_lr_mults'))).astype(float)
    assert len(args.classifier_heads) == len(args.classifier_lr_mults)

    # assert clf_type in ['l2softmax', 'adm', 'adacos', 'xvec', 'arcface', 'sphereface', 'softmax']

    args.classifier_smooth_types = config['Optim'].get('classifier_smooth_types').split(',')
    assert len(args.classifier_smooth_types) == len(args.classifier_heads)
    args.classifier_smooth_types = [s.strip() for s in args.classifier_smooth_types]

    args.label_smooth_type = config['Optim'].get('label_smooth_type', fallback='None')
    assert args.label_smooth_type in ['None', 'disturb', 'uniform']
    args.label_smooth_prob = config['Optim'].getfloat('label_smooth_prob', fallback=0.1)

    args.input_dim = config['Hyperparams'].getint('input_dim', fallback=30)
    args.embedding_dim = config['Hyperparams'].getint('embedding_dim', fallback=512)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)


    args.model_dir = config['Outputs']['model_dir']
    if not hasattr(args, 'basefolder'):
        args.basefolder = config['Outputs'].get('basefolder', fallback=None)
    args.log_file = os.path.join(args.model_dir, 'train.log')
    args.results_pkl = os.path.join(args.model_dir, 'results.p')

    args.num_age_bins = config['Misc'].getint('num_age_bins', fallback=10)
    args.age_label_smoothing = config['Misc'].getboolean('age_label_smoothing', fallback=False)
    return args


def test(generator, ds_test, device, mindcf=False):
    generator.eval()
    all_embeds = []
    all_utts = []
    num_examples = len(ds_test.veri_utts)

    with torch.no_grad():
        for i in range(num_examples):
            feats, utt = ds_test.__getitem__(i)
            feats = feats.unsqueeze(0).to(device)
            embeds = generator(feats)
            all_embeds.append(embeds.cpu().numpy())
            all_utts.append(utt)

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})

    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    eer, mindcf1 = metric.compute_min_cost(scores, 1 - ds_test.veri_labs)
    generator.train()
    if mindcf:
        return eer, mindcf1, None
    else:
        return eer

def test_fromdata(generator, all_feats, all_utts, ds_test, device, mindcf=False):
    generator.eval()
    all_embeds = []

    with torch.no_grad():
        for feats in all_feats: 
            feats = feats.unsqueeze(0).to(device)
            embeds = generator(feats)
            all_embeds.append(embeds.cpu().numpy())

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})

    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    eer, mindcf1 = metric.compute_min_cost(scores, 1 - ds_test.veri_labs)
    generator.train()
    if mindcf:
        return eer, mindcf1, None
    else:
        return eer


def test_all_factors_multids(model_dict, ds_test, dsl, label_types, device):
    assert ds_test.test_mode

    for m in model_dict:
        model_dict[m]['model'].eval()

    label_types = [l for l in label_types if l not in ['speaker', 'rec']]

    with torch.no_grad():
        feats, label_dict, all_utts = ds_test.get_test_items()
        all_embeds = []
        pred_dict = {m: [] for m in label_types}
        for feat in tqdm(feats):
            feat = feat.unsqueeze(0).to(device)
            embed = model_dict['generator']['model'](model_dict['{}_ilayer'.format(dsl)]['model'](feat))
            for m in label_types:
                dictkey = '{}_{}'.format(dsl, m)
                pred = torch.argmax(model_dict[dictkey]['model'](embed, label=None), dim=1)
                pred_dict[m].append(pred.cpu().numpy()[0])
            all_embeds.append(embed.cpu().numpy())

    accuracy_dict = {m: np.equal(label_dict[m], pred_dict[m]).sum() / len(all_utts) for m in label_types}

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})
    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    print('Min score: {}, max score {}'.format(min(scores), max(scores)))
    eer, mindcf1 = metric.compute_min_cost(scores, 1 - ds_test.veri_labs)

    for m in model_dict:
        model_dict[m]['model'].train()

    return eer, mindcf1, accuracy_dict


def test_all_factors(model_dict, ds_test, device):
    assert ds_test.test_mode

    for m in model_dict:
        model_dict[m]['model'].eval()

    label_types = [l for l in ds_test.label_types if l in model_dict]

    with torch.no_grad():
        feats, label_dict, all_utts = ds_test.get_test_items()
        all_embeds = []
        pred_dict = {m: [] for m in label_types}
        for feat in tqdm(feats):
            feat = feat.unsqueeze(0).to(device)
            embed = model_dict['generator']['model'](feat)
            for m in label_types:
                if m.endswith('regression'):
                    pred = model_dict[m]['model'](embed, label=None)
                else:
                    pred = torch.argmax(model_dict[m]['model'](embed, label=None), dim=1)
                pred_dict[m].append(pred.cpu().numpy()[0])
            all_embeds.append(embed.cpu().numpy())

    accuracy_dict = {m: np.equal(label_dict[m], pred_dict[m]).sum() / len(all_utts) for m in label_types if not m.endswith('regression')}

    for m in label_types:
        if m.endswith('regression'):
            accuracy_dict[m] = np.mean((label_dict[m] - pred_dict[m])**2)

    if ds_test.veripairs:
        metric = SpeakerRecognitionMetrics(distance_measure='cosine')
        all_embeds = np.vstack(all_embeds)
        all_embeds = normalize(all_embeds, axis=1)
        all_utts = np.array(all_utts)

        utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})
        emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
        emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

        scores = metric.scores_from_pairs(emb0, emb1)
        print('Min score: {}, max score {}'.format(min(scores), max(scores)))
        eer, mindcf1 = metric.compute_min_cost(scores, 1 - ds_test.veri_labs)
    else:
        eer, mindcf1 = 0.0, 0.0

    for m in model_dict:
        model_dict[m]['model'].train()

    return eer, mindcf1, accuracy_dict


def test_all_factors_ensemble(model_dict, ds_test, device, feats, all_utts, exclude=[], combine='sum'):
    assert ds_test.test_mode

    for m in model_dict:
        model_dict[m]['model'].eval()

    label_types = [l for l in ds_test.label_types if l in model_dict]
    set_veri_utts = set(list(ds_test.veri_0) + list(ds_test.veri_1))
    aux_embeds_dict = {m: [] for m in label_types}

    with torch.no_grad():
        veri_embeds = []
        veri_utts = []
        for feat, utt in tqdm(zip(feats, all_utts)):
            if utt in set_veri_utts:
                feat = feat.unsqueeze(0).to(device)
                embed = model_dict['generator']['model'](feat)
                veri_embeds.append(embed)
                veri_utts.append(utt)
    
        veri_embeds = torch.cat(veri_embeds)
        for m in label_types:
            task_embeds = model_dict[m]['model'](veri_embeds, label=None, transform=True)
            aux_embeds_dict[m] = normalize(task_embeds.cpu().numpy(), axis=1)
    
        veri_embeds = normalize(veri_embeds.cpu().numpy(), axis=1)

    aux_embeds_dict['base'] = veri_embeds

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')

    total_scores = []

    for key in aux_embeds_dict:
        if key not in exclude:
            utt_embed = OrderedDict({k: v for k, v in zip(veri_utts, aux_embeds_dict[key])})
            emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
            emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])
            scores = metric.scores_from_pairs(emb0, emb1)
            total_scores.append(scores)

    if combine == 'sum':
        total_scores = np.sum(np.array(total_scores), axis=0)
        eer, mindcf1 = metric.compute_min_cost(total_scores, 1. - ds_test.veri_labs)
    else:
        total_scores = np.array(total_scores).T
        lr_clf = LogisticRegression(solver='lbfgs')
        lr_clf.fit(total_scores, 1. - ds_test.veri_labs)
        weighted_scores = lr_clf.predict(total_scores)
        eer, mindcf1 = metric.compute_min_cost(weighted_scores, 1. - ds_test.veri_labs)

    for m in model_dict:
        model_dict[m]['model'].train()

    return eer, mindcf1


def test_enrollment_models(generator, ds_test, device, return_scores=False, reduce_method='mean_embed'):
    assert reduce_method in ['mean_embed', 'mean_dist', 'max', 'min']
    generator.eval()
    all_embeds = []
    all_utts = []
    num_examples = len(ds_test)

    with torch.no_grad():
        for i in tqdm(range(num_examples)):
            feats, utt = ds_test.__getitem__(i)
            feats = feats.unsqueeze(0).to(device)
            embeds = generator(feats)
            all_embeds.append(embeds.cpu().numpy())
            all_utts.append(utt)

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})
    model_embeds = OrderedDict({})
    for i, model in enumerate(ds_test.models):
        model_embeds[model] = np.array([utt_embed[utt] for utt in ds_test.m_utts[i]])

    emb0s = np.array([model_embeds[u] for u in ds_test.models_eval])
    emb1 = np.array([utt_embed[u] for u in ds_test.eval_utts])

    mscore_means = []
    mscore_stds = []
    scores = []
    for model_utts, test_utt in zip(emb0s, emb1):
        if reduce_method == 'mean_embed':
            model_std = model_utts.std(0)
            model_mean = model_utts.mean(0)
            scores.append(np.linalg.norm(test_utt - model_mean))
        elif reduce_method == 'mean_dist':
            dist_means = np.mean(np.array([np.linalg.norm(test_utt - e) for e in model_utts]))
            scores.append(dist_means)
        elif reduce_method == 'min':
            scores.append(np.min(np.array([np.linalg.norm(test_utt - e) for e in model_utts])))
        elif reduce_method == 'max':
            scores.append(np.max(np.array([np.linalg.norm(test_utt - e) for e in model_utts])))
        else:
            print('do nothing')

    scores = np.array(scores)
    if return_scores:
        return scores

    eer, mindcf1 = metric.compute_min_cost(scores,
                                           1 - ds_test.veri_labs)
    generator.train()
    return eer, mindcf1, scores


def test_nosil(generator, ds_test, device, mindcf=False):
    generator.eval()
    all_embeds = []
    all_utts = []
    num_examples = len(ds_test.veri_utts)

    with torch.no_grad():
        with ReadHelper(
                'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{0}/feats_trimmed.scp '
                'ark:- | select-voiced-frames ark:- scp:{0}/vad_trimmed.scp ark:- |'.format(
                        ds_test.data_base_path)) as reader:
            for key, feat in tqdm(reader, total=num_examples):
                if key in ds_test.veri_utts:
                    all_utts.append(key)
                    feats = torch.FloatTensor(feat).unsqueeze(0).to(device)
                    embeds = generator(feats)
                    all_embeds.append(embeds.cpu().numpy())

    metric = SpeakerRecognitionMetrics(distance_measure='cosine')
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    print(all_embeds.shape, len(ds_test.veri_utts))
    utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})

    emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
    emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

    scores = metric.scores_from_pairs(emb0, emb1)
    fpr, tpr, thresholds = roc_curve(1 - ds_test.veri_labs, scores, pos_label=1, drop_intermediate=False)
    eer = metric.eer_from_ers(fpr, tpr)
    generator.train()
    if mindcf:
        mindcf1 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.01)
        mindcf2 = metric.compute_min_dcf(fpr, tpr, thresholds, p_target=0.001)
        return eer, mindcf1, mindcf2
    else:
        return eer


def evaluate_deepmine(generator, ds_eval, device, outfile_path='./exp'):
    os.makedirs(outfile_path, exist_ok=True)
    generator.eval()

    answer_col0 = []
    answer_col1 = []
    answer_col2 = []

    with torch.no_grad():
        for i in tqdm(range(len(ds_eval))):
            model, enrol_utts, enrol_feats, eval_utts, eval_feats = ds_eval.__getitem__(i)
            answer_col0.append([model for _ in range(len(eval_utts))])
            answer_col1.append(eval_utts)

            enrol_feats = mtd(enrol_feats, device)
            model_embed = torch.cat([generator(x.unsqueeze(0)) for x in enrol_feats]).cpu().numpy()
            model_embed = np.mean(normalize(model_embed, axis=1), axis=0).reshape(1, -1)

            del enrol_feats
            eval_feats = mtd(eval_feats, device)
            eval_embeds = torch.cat([generator(x.unsqueeze(0)) for x in eval_feats]).cpu().numpy()
            eval_embeds = normalize(eval_embeds, axis=1)

            scores = cosine_similarity(model_embed, eval_embeds).squeeze(0)
            assert len(scores) == len(eval_utts)
            answer_col2.append(scores)
            del eval_feats

    answer_col0 = np.concatenate(answer_col0)
    answer_col1 = np.concatenate(answer_col1)
    answer_col2 = np.concatenate(answer_col2)

    with open(os.path.join(outfile_path, 'answer_full.txt'), 'w+') as fp:
        for m, ev, s in zip(answer_col0, answer_col1, answer_col2):
            line = '{} {} {}\n'.format(m, ev, s)
            fp.write(line)

    with open(os.path.join(outfile_path, 'answer.txt'), 'w+') as fp:
        for s in answer_col2:
            line = '{}\n'.format(s)
            fp.write(line)

    if (answer_col0 == np.array(ds_eval.models_eval)).all():
        print('model ordering matched')
    else:
        print('model ordering was not correct, need to fix before submission')

    if (answer_col1 == np.array(ds_eval.eval_utts)).all():
        print('eval utt ordering matched')
    else:
        print('eval utt ordering was not correct, need to fix before submission')


def dvec_compute(generator, ds_eval, device, num_jobs=20, outfolder='./exp/example_dvecs'):
    # naively compute the embeddings for each window
    # ds_len = len(ds_feats)
    all_utts = ds_eval.all_utts
    ds_len = len(all_utts)
    indices = np.arange(ds_len)
    job_split = np.array_split(indices, num_jobs)
    generator.eval().to(device)
    for job_num, job in enumerate(tqdm(job_split)):
        print('Starting job {}'.format(job_num))
        ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{0}/xvector.{1}.ark,{0}/xvector.{1}.scp'.format(outfolder,
                                                                                                          job_num + 1)
        job_utts = all_utts[job]
        job_feats = ds_eval.get_batches(job_utts)
        job_feats = mtd(job_feats, device)
        with torch.no_grad():
            job_embeds = torch.cat([generator(x.unsqueeze(0)) for x in tqdm(job_feats)]).cpu().numpy()
        with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
            for xvec, key in zip(job_embeds, job_utts):
                kaldi_io.write_vec_flt(f, xvec, key=key)


def evaluate_deepmine_from_xvecs(ds_eval, outfolder='./exp/example_xvecs'):
    if not os.path.isfile(os.path.join(outfolder, 'xvector.scp')):
        xvec_scps = glob(os.path.join(outfolder, '*.scp'))
        assert len(xvec_scps) != 0, 'No xvector scps found'
        with open(os.path.join(outfolder, 'xvector.scp'), 'w+') as outfile:
            for fname in xvec_scps:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

    xvec_dict = odict_from_2_col(os.path.join(outfolder, 'xvector.scp'))
    answer_col0 = []
    answer_col1 = []
    answer_col2 = []

    for i in tqdm(range(len(ds_eval))):
        model, enrol_utts, eval_utts, = ds_eval.get_item_utts(i)
        answer_col0.append([model for _ in range(len(eval_utts))])
        answer_col1.append(eval_utts)

        model_embeds = np.array([read_vec_flt(xvec_dict[u]) for u in enrol_utts])
        model_embed = np.mean(normalize(model_embeds, axis=1), axis=0).reshape(1, -1)

        eval_embeds = np.array([read_vec_flt(xvec_dict[u]) for u in eval_utts])
        eval_embeds = normalize(eval_embeds, axis=1)

        scores = cosine_similarity(model_embed, eval_embeds).squeeze(0)
        assert len(scores) == len(eval_utts)
        answer_col2.append(scores)

    answer_col0 = np.concatenate(answer_col0)
    answer_col1 = np.concatenate(answer_col1)
    answer_col2 = np.concatenate(answer_col2)

    print('Writing results to file...')
    with open(os.path.join(outfolder, 'answer_full.txt'), 'w+') as fp:
        for m, ev, s in tqdm(zip(answer_col0, answer_col1, answer_col2)):
            line = '{} {} {}\n'.format(m, ev, s)
            fp.write(line)

    with open(os.path.join(outfolder, 'answer.txt'), 'w+') as fp:
        for s in tqdm(answer_col2):
            line = '{}\n'.format(s)
            fp.write(line)

    if (answer_col0 == np.array(ds_eval.models_eval)).all():
        print('model ordering matched')
    else:
        print('model ordering was not correct, need to fix before submission')

    if (answer_col1 == np.array(ds_eval.eval_utts)).all():
        print('eval utt ordering matched')
    else:
        print('eval utt ordering was not correct, need to fix before submission')


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def get_eer_metrics(folder):
    rpkl_path = os.path.join(folder, 'wrec_results.p')
    rpkl = pickle.load(open(rpkl_path, 'rb'))
    iterations = list(rpkl.keys())
    eers = [rpkl[k]['test_eer'] for k in rpkl]
    return iterations, eers, np.min(eers)


def extract_test_embeds(generator, ds_test, device):
    assert ds_test.test_mode

    with torch.no_grad():
        feats, label_dict, all_utts = ds_test.get_test_items()
        all_embeds = []
        for feat in tqdm(feats):
            feat = feat.unsqueeze(0).to(device)
            embed = generator(feat)
            all_embeds.append(embed.cpu().numpy())

    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)
    all_utts = np.array(all_utts)

    return all_utts, all_embeds

if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    uvloop.install()

    hf = h5py.File(os.path.join(args.model_dir, 'xvectors.h5'), 'w')

    use_cuda = torch.cuda.is_available()
    print('=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.checkpoint == -1:
        g_path = os.path.join(args.model_dir, "final_generator_{}.pt".format(args.num_iterations))
    else:
        g_path = os.path.join(args.model_dir, "generator_{}.pt".format(args.checkpoint))

    if args.model_type == 'XTDNN':
        generator = XTDNN(features_per_frame=args.input_dim, embed_features=args.embedding_dim)
    if args.model_type == 'ETDNN':
        generator = ETDNN(features_per_frame=args.input_dim, embed_features=args.embedding_dim)
    if args.model_type == 'FTDNN':
        generator = FTDNN(in_dim=args.input_dim, embedding_dim=args.embedding_dim)

    if args.best:
        iterations, eers, _ = get_eer_metrics(args.model_dir)
        best_it = iterations[np.argmin(eers)]
        g_path = os.path.join(args.model_dir, "generator_{}.pt".format(best_it))
    
    assert os.path.isfile(g_path), "Couldn't find {}".format(g_path)

    ds_train = SpeakerDataset(args.train_data, num_age_bins=args.num_age_bins)
    class_enc_dict = ds_train.get_class_encs()
    if args.test_data:
        ds_test = SpeakerDataset(args.test_data, test_mode=True, 
                                    class_enc_dict=class_enc_dict, num_age_bins=args.num_age_bins)

    generator.load_state_dict(torch.load(g_path))
    model = generator
    model.eval().to(device)

    utts, embeds = extract_test_embeds(generator, ds_test, device)

    hf.create_dataset('embeds', data=embeds)
    dt = h5py.string_dtype()
    hf.create_dataset('utts', data=np.string_(np.array(utts)), dtype=dt)
    hf.close()
   
