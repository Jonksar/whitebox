import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from sklearn.base import ClassifierMixin, TransformerMixin, RegressorMixin
from funcsigs import signature

from .utils import eprint


class ClassifierPlots:
    def __init__(self, clf, data, labels, cutoff=0.5):
        if not isinstance(ClassifierMixin, clf):
            raise ValueError("Classifier must be subclassed from sklearn.ClassifierMixin")

        self._clf = clf
        self.X = data
        self.y = labels
        self.predictions   = self._clf.predict(self.X)
        self.predictions_t = self.predictions > cutoff

        self.cutoff = cutoff

    def _plot_pr(self, label_name="label", cutoff=0.5):

        precision, recall, _ = precision_recall_curve(self.y, self.predictions)

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

        average_precision = average_precision_score(self.y, self.predictions)
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))

    def _plot_threshold(self, label_name="label", cutoff=0.5):
        """Show precision and recall as a function of different decision thresholds."""
        precision, recall, thresholds = precision_recall_curve(self.y, self.predictions, pos_label=1)

        plot_p, = plt.plot(thresholds, precision[1:], "--", label="precision")j
        plot_r, = plt.plot(thresholds, recall[1:], label="recall")
        plt.xlabel('Confidence')
        plt.ylabel('Metric')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()

    def _plot_histograms(self, label_name="label", cutoff=0.5):
        labels = self.y.copy()

        if labels.dtype == 'bool':
            p_label = "positive" + " size: %d" % np.sum(labels)
            n_label = "negative" + " size: %d" % np.sum(np.logical_not(labels))

        elif labels.dtype == 'float' or labels.dtype == 'int':
            p_label = label_name + " positive" + (
                "" if cutoff is None else "at cutoff: " + str(cutoff)) + " size: %d" % np.sum(labels)
            n_label = label_name + " negative" + (
                "" if cutoff is None else "at cutoff: " + str(cutoff)) + " size: %d" % np.sum(np.logical_not(labels))

            labels = labels > cutoff
        else:
            raise ValueError("Unknown input self.X format %s" % labels.dtype)

        _range = (float(np.min(self.X)), float(np.max(self.X)))

        try:
            ax = sns.distplot(self.X[labels], label=p_label, hist_kws={"range": _range}, bins=50)
        except Exception as e:
            eprint(str(e))
        try:
            ax = sns.distplot(self.X[~labels], label=n_label, hist_kws={"range": _range}, bins=50)
        except Exception as e:
            eprint(str(e))
        plt.legend()
        plt.title("Confidence distributions on positive & negative classes (n=%d)" % len(self.X))

    def plot(self, **kwargs):
        fig1 = plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 3, 1)
        self._plot_histograms(label_name=kwargs.get('label_name', None), cutoff=kwargs.get('cutoff', 0.5))

        plt.subplot(1, 3, 2)
        self._plot_threshold(label_name=kwargs.get('label_name', None), cutoff=kwargs.get('cutoff', 0.5))

        plt.subplot(1, 3, 3)
        self._plot_pr(label_name=kwargs.get('label_name', None), cutoff=kwargs.get('cutoff', 0.5))

    def show(self):
        plt.show()
