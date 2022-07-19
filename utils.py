import xml.etree.ElementTree as ET
import tqdm
import numpy as np
import pandas as pd
import time
from rdkit import Chem
import gc
import pickle
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import random
from keras import models, layers, losses, utils, callbacks, metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import snf_simple
from sklearn.metrics import roc_curve, auc
from snf import compute
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ns = "{http://www.drugbank.ca}"  # the website of DrugBank
dataset_file = "./data/drugs/full database.xml"  # the file path of downloaded DrugBank dataset


def performance_helper(test_num, y_real, y_pred):
    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative

    for i in range(test_num):
        if y_real[i] == 1 and y_pred[i] == y_real[i]:
            tp = tp + 1
        elif y_real[i] == 1 and y_pred[i] != y_real[i]:
            fn = fn + 1
        elif y_real[i] == 0 and y_pred[i] == y_real[i]:
            tn = tn + 1
        else:
            fp = fp + 1

    print(tp, tn, fp, fn)
    # performance evaluation measures
    _acc = float(tp + tn) / (tp + tn + fp + fn)
    _recall = tp / float(tp + fn)  # _sensitivity
    _precision = tp / float(tp + fp)
    _specificity = tn / float(tn + fp)
    _f1score = (2 * _precision * _recall) / (_precision + _recall)

    return _acc, _recall, _precision, _specificity, _f1score


class SmallMolecules:
    def __init__(self, index, name, **kw):
        self.id = index
        self.name = name
        for k, w in kw.items():
            setattr(self, k, w)

