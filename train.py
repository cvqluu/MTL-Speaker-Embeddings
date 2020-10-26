import argparse
import configparser
import glob
import json
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from pprint import pprint
from inference import test, test_all_factors, test_enrollment_models, test_nosil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvloop
from data_io import SpeakerDataset, SpeakerModelTestDataset, SpeakerTestDataset
from models.classifiers import (AdaCos, AMSMLoss, ArcFace, L2SoftMax, SoftMax,
                                SphereFace, XVecHead, XVecHeadUncertain, GradientReversal)
from models.criteria import (DisturbLabelLoss, LabelSmoothingLoss,
                             TwoNeighbourSmoothingLoss, MultiTaskUncertaintyLossKendall, 
                             MultiTaskUncertaintyLossLiebel)
from models.extractors import ETDNN, FTDNN, XTDNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import SpeakerRecognitionMetrics, schedule_lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description='Train SV model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--transfer-learning', action='store_true', default=False,
                        help='Start from g_start.pt in exp folder')
    parser.add_argument('--resume-checkpoint', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
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
        assert clf in ['speaker', 'nationality', 'gender', 'age', 'age_regression', 'rec']

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
    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.2)
    args.batch_size = config['Hyperparams'].getint('batch_size', fallback=400)
    args.max_seq_len = config['Hyperparams'].getint('max_seq_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)
    args.momentum = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.scheduler_steps = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.5)
    args.multi_gpu = config['Hyperparams'].getboolean('multi_gpu', fallback=False)
    args.classifier_lr_mult = config['Hyperparams'].getfloat('classifier_lr_mult', fallback=1.)
    args.dropout = config['Hyperparams'].getboolean('dropout', fallback=True)

    args.model_dir = config['Outputs']['model_dir']
    if not hasattr(args, 'basefolder'):
        args.basefolder = config['Outputs'].get('basefolder', fallback=None)
    args.log_file = os.path.join(args.model_dir, 'train.log')
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')
    args.results_pkl = os.path.join(args.model_dir, 'results.p')

    args.num_age_bins = config['Misc'].getint('num_age_bins', fallback=10)
    args.age_label_smoothing = config['Misc'].getboolean('age_label_smoothing', fallback=False)
    return args


def train(ds_train):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30)
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(comment=os.path.basename(args.cfg))

    if args.model_type == 'XTDNN':
        generator = XTDNN(features_per_frame=args.input_dim, embed_features=args.embedding_dim)
    if args.model_type == 'ETDNN':
        generator = ETDNN(features_per_frame=args.input_dim, embed_features=args.embedding_dim)
    if args.model_type == 'FTDNN':
        generator = FTDNN(in_dim=args.input_dim, embedding_dim=args.embedding_dim)

    generator.train()
    generator = generator.to(device)

    model_dict = {'generator': {'model': generator, 'lr_mult': 1., 'loss_weight': None}}
    clf_head_dict = {k: {'model': None, 'lr_mult': lr_mult, 'loss_weight': loss_weight} for k, lr_mult, loss_weight in
                     zip(args.classifier_heads, args.classifier_lr_mults, args.classifier_loss_weights)}

    num_cls_per_task = [ds_train.num_classes[t] for t in args.classifier_heads]

    for clf_target, clf_type, num_classes, clf_smooth_type in zip(args.classifier_heads, 
                                                                    args.classifier_types, 
                                                                    num_cls_per_task, 
                                                                    args.classifier_smooth_types):
        if clf_type == 'adm':
            clf = AMSMLoss(args.embedding_dim, num_classes)
        elif clf_type == 'adacos':
            clf = AdaCos(args.embedding_dim, num_classes)
        elif clf_type == 'l2softmax':
            clf = L2SoftMax(args.embedding_dim, num_classes)
        elif clf_type == 'softmax':
            clf = SoftMax(args.embedding_dim, num_classes)
        elif clf_type == 'xvec':
            clf = XVecHead(args.embedding_dim, num_classes)
        elif clf_type == 'xvec_regression':
            clf = XVecHead(args.embedding_dim, 1)
        elif clf_type == 'xvec_uncertain':
            clf = XVecHeadUncertain(args.embedding_dim, num_classes)
        elif clf_type == 'arcface':
            clf = ArcFace(args.embedding_dim, num_classes)
        elif clf_type == 'sphereface':
            clf = SphereFace(args.embedding_dim, num_classes)
        else:
            assert None, 'Classifier type {} not found'.format(clf_type)

        if clf_head_dict[clf_target]['loss_weight'] >= 0.0:
            clf_head_dict[clf_target]['model'] = clf.train().to(device)
        else:
            # GRL for negative loss weight
            abs_lw = np.abs(clf_head_dict[clf_target]['loss_weight'])
            clf_head_dict[clf_target]['model'] = nn.Sequential(
                                                                GradientReversal(lambda_=abs_lw),
                                                                clf
                                                                ).train().to(device)
            clf_head_dict[clf_target]['loss_weight'] = 1.0 # this is lambda_ in the GRL

        if clf_smooth_type == 'none':
            if clf_target.endswith('regression'):
                clf_smooth = nn.SmoothL1Loss()
            else:
                clf_smooth = nn.CrossEntropyLoss()
        elif clf_smooth_type == 'twoneighbour':
            clf_smooth = TwoNeighbourSmoothingLoss(smoothing=args.label_smooth_prob)
        elif clf_smooth_type == 'uniform':
            clf_smooth = LabelSmoothingLoss(smoothing=args.label_smooth_prob)
        elif clf_smooth_type == 'disturb':
            clf_smooth = DisturbLabelLoss(device, disturb_prob=args.label_smooth_prob)
        else:
            assert None, 'Smooth type not found: {}'.format(clf_smooth_type)
        
        clf_head_dict[clf_target]['criterion'] = clf_smooth

    model_dict.update(clf_head_dict)

    if args.classifier_loss_weighting_type == 'uncertainty_kendall':
        model_dict['loss_aggregator'] = {
                                            'model': MultiTaskUncertaintyLossKendall(len(args.classifier_heads)).to(device),
                                            'lr_mult': 1.,
                                            'loss_weight': None
                                        }
    if args.classifier_loss_weighting_type == 'uncertainty_liebel':
        model_dict['loss_aggregator'] = {
                                            'model': MultiTaskUncertaintyLossLiebel(len(args.classifier_heads)).to(device),
                                            'lr_mult': 1.,
                                            'loss_weight': None
                                        }

    if args.resume_checkpoint != 0:
        model_str = os.path.join(args.model_dir, '{}_{}.pt')
        for m in model_dict:
            model_dict[m]['model'].load_state_dict(torch.load(model_str.format(m, args.resume_checkpoint)))

    optimizer = torch.optim.SGD(
        [{'params': model_dict[m]['model'].parameters(), 'lr': args.lr * model_dict[m]['lr_mult']} for m in model_dict],
        momentum=args.momentum)


    iterations = 0

    total_loss = 0
    running_loss = [np.nan for _ in range(500)]

    non_spk_clf_heads = [a for a in args.classifier_heads if a != 'speaker']
    
    best_test_eer = (-1, 1.0)
    best_test_dcf = (-1, 1.0)
    best_acc = {k: (-1, 0.0) for k in non_spk_clf_heads}

    if os.path.isfile(args.results_pkl) and args.resume_checkpoint != 0:
        rpkl = pickle.load(open(args.results_pkl, "rb"))
        keylist = list(rpkl.keys())

        if args.test_data:
            test_eers = [(rpkl[key]['test_eer'], key) for i, key in enumerate(rpkl)]
            best_teer = min(test_eers)
            best_test_eer = (best_teer[1], best_teer[0])

            test_dcfs = [(rpkl[key]['test_dcf'], key) for i, key in enumerate(rpkl)]
            besttest_dcf = min(test_dcfs)
            best_test_dcf = (besttest_dcf[1], besttest_dcf[0])

    else:
        rpkl = OrderedDict({})

    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)

    data_generator = ds_train.get_batches(batch_size=args.batch_size, max_seq_len=args.max_seq_len)

    if args.model_type == 'FTDNN':
        drop_indexes = np.linspace(0, 1, args.num_iterations)
        drop_sch = ([0, 0.5, 1], [0, 0.5, 0])
        drop_schedule = np.interp(drop_indexes, drop_sch[0], drop_sch[1])

    for iterations in range(1, args.num_iterations + 1):
        if iterations > args.num_iterations:
            break
        if iterations in args.scheduler_steps:
            schedule_lr(optimizer, factor=args.scheduler_lambda)
        if iterations <= args.resume_checkpoint:
            print('Skipping iteration {}'.format(iterations), file=open(args.log_file, "a"))
            continue

        if args.model_type == 'FTDNN':
            if args.dropout:
                generator.set_dropout_alpha(drop_schedule[iterations - 1])

        feats, labels = next(data_generator)
        feats = feats.to(device)

        if args.multi_gpu:
            embeds = dpp_generator(feats)
        else:
            embeds = generator(feats)

        total_loss = 0
        losses = []

        loss_tensors = []

        for m in args.classifier_heads:
            lab = labels[m].to(device)
            if m == 'rec':
                preds = model_dict[m]['model'](embeds)
            else:
                preds = model_dict[m]['model'](embeds, lab)
            loss = model_dict[m]['criterion'](preds, lab)
            if args.classifier_loss_weighting_type == 'none':
                total_loss += loss * model_dict[m]['loss_weight']
            else:
                loss_tensors.append(loss)
            losses.append(round(loss.item(), 4))

        if args.classifier_loss_weighting_type.startswith('uncertainty'):
            loss_tensors = torch.FloatTensor(loss_tensors).to(device)
            total_loss = model_dict['loss_aggregator']['model'](loss_tensors)

        if args.classifier_loss_weighting_type == 'dwa':
            loss_tensors = loss_tensors
            if iterations < 4:
                loss_t_1 = np.ones(len(loss_tensors))
                for l in loss_tensors:
                    total_loss += l
            else:
                dwa_w = loss_t_1/loss_t_2
                K = len(loss_tensors)
                per_task_weight = torch.FloatTensor(dwa_w/args.dwa_temperature) #lambda_k
                per_task_weight = torch.nn.functional.softmax(per_task_weight, dim=0) * K
                per_task_weight = per_task_weight.numpy()
                for l, w in zip(loss_tensors, per_task_weight):
                    total_loss += l * w

            loss_t_2 = loss_t_1.copy()
            loss_t_1 = torch.FloatTensor(loss_tensors).detach().cpu().numpy()

            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if args.model_type == 'FTDNN':
            generator.step_ftdnn_layers()

        running_loss.pop(0)
        running_loss.append(total_loss.item())
        rmean_loss = np.nanmean(np.array(running_loss))

        if iterations % 10 == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, AvgLoss:{:.4f}, losses: {}, lr: {}, bs: {}".format(
                args.model_dir,
                time.ctime(),
                iterations,
                args.num_iterations,
                total_loss.item(),
                rmean_loss,
                losses,
                get_lr(optimizer),
                len(feats))
            print(msg)
            print(msg, file=open(args.log_file, "a"))

        writer.add_scalar('combined loss', total_loss.item(), iterations)
        writer.add_scalar('Avg loss', rmean_loss, iterations)

        if iterations % args.checkpoint_interval == 0:
            for m in model_dict:
                model_dict[m]['model'].eval().cpu()
                cp_filename = "{}_{}.pt".format(m, iterations)
                cp_model_path = os.path.join(args.model_dir, cp_filename)
                torch.save(model_dict[m]['model'].state_dict(), cp_model_path)
                model_dict[m]['model'].to(device).train()

            if args.test_data:
                rpkl, best_test_eer, best_test_dcf = eval_step(model_dict, device, ds_test, iterations, rpkl, writer,
                                                           best_test_eer, best_test_dcf, best_acc)

    # ---- Final model saving -----
    for m in model_dict:
        model_dict[m]['model'].eval().cpu()
        cp_filename = "final_{}_{}.pt".format(m, iterations)
        cp_model_path = os.path.join(args.model_dir, cp_filename)
        torch.save(model_dict[m]['model'].state_dict(), cp_model_path)


def eval_step(model_dict, device, ds_test, iterations, rpkl, writer, best_test_eer, best_test_dcf, best_acc):
    rpkl[iterations] = {}
    print('Evaluating on test/validation for {} iterations'.format(iterations))
    if args.test_data:
        test_eer, test_dcf, acc_dict = test_all_factors(model_dict, ds_test, device)
        print(acc_dict)
        for att in acc_dict:
            writer.add_scalar(att, acc_dict[att], iterations)
            if acc_dict[att] > best_acc[att][1]:
                best_acc[att] = (iterations, acc_dict[att])
            print('{} accuracy on Test Set: {}'.format(att, acc_dict[att]))
            print('{} accuracy on Test Set: {}'.format(att, acc_dict[att]), file=open(args.log_file, "a"))
            print('Best test {} acc: {}'.format(att, best_acc[att]))
            print('Best test {} acc: {}'.format(att, best_acc[att]), file=open(args.log_file, "a"))
            rpkl[iterations][att] = acc_dict[att]

        print('EER on Test Set: {} ({})'.format(test_eer, args.test_data))
        print('EER on Test Set: {} ({})'.format(test_eer, args.test_data), file=open(args.log_file, "a"))
        writer.add_scalar('test_eer', test_eer, iterations)
        if test_eer < best_test_eer[1]:
            best_test_eer = (iterations, test_eer)
        print('Best test EER: {}'.format(best_test_eer))
        print('Best test EER: {}'.format(best_test_eer), file=open(args.log_file, "a"))
        rpkl[iterations]['test_eer'] = test_eer

        print('minDCF on Test Set: {} ({})'.format(test_dcf, args.test_data))
        print('minDCF on Test Set: {} ({})'.format(test_dcf, args.test_data), file=open(args.log_file, "a"))
        writer.add_scalar('test_dcf', test_dcf, iterations)
        if test_dcf < best_test_dcf[1]:
            best_test_dcf = (iterations, test_dcf)
        print('Best test minDCF: {}'.format(best_test_dcf))
        print('Best test minDCF: {}'.format(best_test_dcf), file=open(args.log_file, "a"))
        rpkl[iterations]['test_dcf'] = test_dcf

    pickle.dump(rpkl, open(args.results_pkl, "wb"))
    return rpkl, best_test_eer, best_test_dcf


if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    os.makedirs(args.model_dir, exist_ok=True)
    if args.resume_checkpoint == 0:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings.cfg'))
    else:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings_resume.cfg'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    pprint(vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    uvloop.install()
    ds_train = SpeakerDataset(args.train_data, num_age_bins=args.num_age_bins)
    class_enc_dict = ds_train.get_class_encs()
    if args.test_data:
        ds_test = SpeakerDataset(args.test_data, test_mode=True, 
                                    class_enc_dict=class_enc_dict, num_age_bins=args.num_age_bins)
    train(ds_train)
