from operator import itemgetter
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import pairwise_distances, roc_curve, accuracy_score
from sklearn.metrics.pairwise import paired_distances
import numpy as np

import torch
import torch.nn as nn


def mtd(stuff, device):
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    else:
        return [mtd(s, device) for s in stuff]

class SpeakerRecognitionMetrics:
    '''
    This doesn't need to be a class [remnant of old structuring]. 
    To be reworked
    '''

    def __init__(self, distance_measure=None):
        if not distance_measure:
            distance_measure = 'cosine'
        self.distance_measure = distance_measure

    def get_labels_scores(self, vectors, labels):
        labels = labels[:, np.newaxis]
        pair_labels = pairwise_distances(labels, metric='hamming').astype(int).flatten()
        pair_scores = pairwise_distances(vectors, metric=self.distance_measure).flatten()
        return pair_labels, pair_scores

    def get_roc(self, vectors, labels):
        pair_labels, pair_scores = self.get_labels_scores(vectors, labels)
        fpr, tpr, threshold = roc_curve(pair_labels, pair_scores, pos_label=1, drop_intermediate=False)
        # fnr = 1. - tpr
        return fpr, tpr, threshold

    def get_eer(self, vectors, labels):
        fpr, tpr, _ = self.get_roc(vectors, labels)
        # fnr = 1 - self.tpr
        # eer = self.fpr[np.nanargmin(np.absolute((fnr - self.fpr)))]
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer

    def eer_from_pairs(self, pair_labels, pair_scores):
        self.fpr, self.tpr, self.thresholds = roc_curve(pair_labels, pair_scores, pos_label=1, drop_intermediate=False)
        fnr = 1 - self.tpr
        eer = self.fpr[np.nanargmin(np.absolute((fnr - self.fpr)))]
        return eer

    def eer_from_ers(self, fpr, tpr):
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer

    def scores_from_pairs(self, vecs0, vecs1):
        return paired_distances(vecs0, vecs1, metric=self.distance_measure)

    def compute_min_dcf(self, fpr, tpr, thresholds, p_target=0.01, c_miss=10, c_fa=1):
        #adapted from compute_min_dcf.py in kaldi sid
        # thresholds, fpr, tpr = list(zip(*sorted(zip(thresholds, fpr, tpr))))
        incr_score_indices = np.argsort(thresholds, kind="mergesort")
        thresholds = thresholds[incr_score_indices]
        fpr = fpr[incr_score_indices]
        tpr = tpr[incr_score_indices]

        fnr = 1. - tpr
        min_c_det = float("inf")
        for i in range(0, len(fnr)):
            c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det

        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf

    def compute_eer(self, fnr, fpr):
        """ computes the equal error rate (EER) given FNR and FPR values calculated
            for a range of operating points on the DET curve
        """

        diff_pm_fa = fnr - fpr
        x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
        x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
        a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
        return fnr[x1] + a * (fnr[x2] - fnr[x1])

    def compute_pmiss_pfa(self, scores, labels):
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their labels. A weights option is also provided
        to equalize the counts over score partitions (if there is such
        partitioning).
        """

        sorted_ndx = np.argsort(scores)
        labels = labels[sorted_ndx]

        tgt = (labels == 1).astype('f8')
        imp = (labels == 0).astype('f8')

        fnr = np.cumsum(tgt) / np.sum(tgt)
        fpr = 1 - np.cumsum(imp) / np.sum(imp)
        return fnr, fpr


    def compute_min_cost(self, scores, labels, p_target=0.01):
        fnr, fpr = self.compute_pmiss_pfa(scores, labels)
        eer = self.compute_eer(fnr, fpr)
        min_c = self.compute_c_norm(fnr, fpr, p_target)
        return eer, min_c

    def compute_c_norm(self, fnr, fpr, p_target, c_miss=10, c_fa=1):
        """ computes normalized minimum detection cost function (DCF) given
            the costs for false accepts and false rejects as well as a priori
            probability for target speakers
        """

        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        c_det = np.min(dcf)
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))

        return c_det/c_def



def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

def schedule_lr(optimizer, factor=0.1):
    for params in optimizer.param_groups:
        params['lr'] *= factor
    print(optimizer)

def set_lr(optimizer, lr):
    for params in optimizer.param_groups:
        params['lr'] = lr
    print(optimizer)


