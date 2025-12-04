import os
import numpy as np
import tensorflow as tf
import csv
from sklearn.metrics import roc_curve
from .model.lcnn import build_lcnn

p_target = 0.01
c_miss = 1
c_fa = 10

# ==== Metrics ====
def calculate_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)
    fnr = 1 - tpr
    eer = fpr[np.argmin(np.abs(fnr - fpr))]
    return eer

def compute_tdcf(y_true, y_score, p_target, c_miss, c_fa):
    thresholds = np.linspace(min(y_score), max(y_score), num=1000)
    tdcf = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        fa = np.sum((y_score >= threshold) & (y_true == 1))
        miss = np.sum((y_score < threshold) & (y_true == 0))
        fa_norm = fa / np.sum(y_true == 1)
        miss_norm = miss / np.sum(y_true == 0)
        tdcf[i] = p_target * c_miss * miss_norm + (1 - p_target) * c_fa * fa_norm

    return tdcf, thresholds  
