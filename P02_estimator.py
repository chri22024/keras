#!/usr/bin/env python3

import os
import pickle


from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import P10_util as util

class Estimator:

    def __init__(self, src_dir, dst_dir, est_file, cls_file, drs_file, srs_file, input_size):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.est_dir = est_dir
        self.cls_file = cls_file
        self.drs_file = drs_file
        self.srs_file = srs_file
        self.input_size = input_size



    def execute(self):

        estimator = load_model(self.est_file)


        with open(self.cls_file,'rb'):
            cls_info = pickle.load(f)


        pred_labels, true_labels, output = [], [], []


        for subdir in os.listdir(self.src_dir):
            for f in os.listdir(os.path.join(self.src_dir, subdir)):
                filename = os.path.join(self.src_dir, subdir, f)
                img = util.load_target_img(filename, self.input_size)


                pred_class = np.argmax(estimator.predict(img))
                pred_label = cls_info[pred_label]
                pred_labels.append(pred_label)


                true_label = subdir
                true_labels.append(true_label)


                output.append('%s -> %s\n' % (filename, pred_label))


        report = classification_report(true_labels, pred_labels)
        labels = list(cls_info.values())
        cnfmtx = confusion_matrix(true_labels, pred_labels, labels)
        cm = pd.DataFrame(cnfmx, index = labels, columns = labels)


        util.mkdir(self.dst_dir, rm = True)
        with open(self.drs_file, 'w') as f:
            f.writelines(output)

        with open(self.srs_file, 'w') as f:
            f.write(report)
            f.write('\n\n')
            f.write(str(cm))
            f.write('\n')
