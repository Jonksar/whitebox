# -*- coding: utf-8 -*-
"""
  --------------------------------------------------
  File Name : reporting.py
  Creation Date : 2018-11-07 T  11:31
  Last Modified : 2018-11-07 T  01:45 
  Created By : Joonatan Samuel
  --------------------------------------------------
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from termcolor import cprint

import matplotlib.pylab as plt
from imageio import imread
import time

from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

def eprint(*args, **kwargs):
    cprint(*args, file=sys.stderr, **kwargs)


def cut_dataset( df, feature_column=None, label_column=None, cutoff=None):

    positive_df  = df[ df[label_column] ]
    negative_df  = df[ np.logical_not( df[label_column] )  ]

    return positive_df, negative_df


def classifier_full_report( df, feature_column='confidence', label_column='label', cutoff=None, dropna=False, display_lambda=None):
    if dropna: df = df.dropna( subset=[ feature_column, label_column] )
    positive_df, negative_df = cut_dataset( df, feature_column, label_column )

    #
    #               Aggregate metric plots
    #

    # Plot 1: Confidence distributions
    if df[label_column].dtype == 'bool':
        p_label = "positive" + " size: %d" % len(positive_df)
        n_label = "negative" + " size: %d" % len(negative_df)
    elif df[label_column].dtype == 'float' or df[label_column].dtype == 'int':
        p_label = label_column + " positive" + ("" if cutoff is None else "at cutoff: " + str(cutoff)) + " size: %d" % len(positive_df)
        n_label = label_column + " negative" + ("" if cutoff is None else "at cutoff: " + str(cutoff)) + " size: %d" % len(negative_df)

    fig1 = plt.figure( figsize=(16, 8))

    plt.subplot(1, 3, 1)
    _range = ( float(np.min( df[feature_column])), float(np.max(df[feature_column])) )
    try: ax = sns.distplot( positive_df[ feature_column ],  label=p_label, hist_kws={"range": _range}, bins=50 )
    except Exception as e: eprint( str(e) )
    try: ax = sns.distplot( negative_df[ feature_column ],  label=n_label, hist_kws={"range": _range}, bins=50 )
    except Exception as e: eprint( str(e) )
    plt.legend()
    plt.title( "Confidence distributions on positive & negative classes (n=%d)" % len(df) )

    # Plot 2: Precision recall curve
    plt.subplot(1, 3, 2)
    precision, recall, _ = precision_recall_curve( df[label_column], df[feature_column])

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')

    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    average_precision = average_precision_score( df[label_column], df[feature_column])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))

    # Plot 3: Confidence & Precision-recall curves
    plt.subplot(1, 3, 3)
    """Show precision and recall as a function of different decision thresholds."""
    precision, recall, thresholds = precision_recall_curve( df[label_column], df[feature_column], pos_label=1)

    plot_p,  = plt.plot(thresholds, precision[1:], "--", label="precision")
    plot_r,  = plt.plot(thresholds, recall[1:], label="recall")
    plt.xlabel('Confidence')
    plt.ylabel('Metric')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()

    # Showing examples of dataset
    print( "Now showing random datapoints from dataset: ")
    for _, row in df.sample(10).iterrows():
        display_lambda(row)
        plt.title('confidence: %6.3f, label %d' % (row[feature_column], row[label_column]))
        plt.show()

    if cutoff is None:
        print( 'cutoff point was not given, defaulting to 0.5')
        cutoff = 0.5

    predictions = df[feature_column] > cutoff

    FP_mask = np.logical_and( np.logical_not(df[label_column]),                predictions )
    TP_mask = np.logical_and(                df[label_column] ,                predictions )
    FN_mask = np.logical_and(                df[label_column] , np.logical_not(predictions))
    TN_mask = np.logical_and( np.logical_not(df[label_column]), np.logical_not(predictions) )

    print( "Now showing FP predictions from dataset: ")
    for _, row in df[ FP_mask ][:10].iterrows():
        display_lambda(row)
        plt.title(' FP (confidence: %6.3f, label %d)' % (row[feature_column], row[label_column]))
        plt.show()

    print( "Now showing FN predictions from dataset: ")
    for _, row in df[ FN_mask ][:10].iterrows():
        display_lambda(row)
        plt.title('FN (confidence: %6.3f, label %d)' % (row[feature_column], row[label_column]))
        plt.show()

    print( "Now showing TN predictions from dataset: ")
    for _, row in df[ TN_mask ][:10].iterrows():
        display_lambda(row)
        plt.title(' TN (confidence: %6.3f, label %d)' % (row[feature_column], row[label_column]))
        plt.show()

    print( "Now showing TP predictions from dataset: ")
    for _, row in df[ TP_mask ][:10].iterrows():
        display_lambda(row)
        plt.title(' TP (confidence: %6.3f, label %d)' % (row[feature_column], row[label_column]))
        plt.show()

    return

if __name__ == "__main__":
    num_dp = 100
    confidences = np.append(
        np.random.normal(0.3, 0.2, num_dp).flatten(),
        np.random.normal(0.7, 0.2, num_dp).flatten() , axis=0
        )

    labels = np.append(
        [ 0 for _ in range(num_dp)],
        [ 1 for _ in range(num_dp)], axis=0
        )

    random_value = np.random.randint(0, 10000, num_dp * 2)

    labels = labels.astype(bool)
    report_df = pd.dataframe({'confidence': confidences, 'label': labels, 'dp_data': random_value})

    def display_image( row ):
        np.random.seed( row['dp_data'] )
        img = np.random.rand( 100, 100 )
        plt.imshow( img )

    classifier_full_report( report_df, display_lambda=display_image )



