import argparse
import csv
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools

from . import underline


class HistoryTracker:
    def __init__(self, save_path=None):
        self.history = defaultdict(list)
        self.learning_rate = None
        self.save_path = save_path
        self.is_train = True

    def start_new_epoch(self, lr):
        self.history.clear()
        self.learning_rate = lr

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def step(self, metrics):
        reports = list()
        for k, v in metrics.items():
            k = k if self.is_train else f'val_{k}'
            if k == "iou_class" or k == "dice_class":
                v = list(itertools.chain.from_iterable(v))
                self.history[k].extend(v)
                k_name = k
                k_list = []
                for i in range(len(v)):
                    k_list.append(k_name+str(i))
                for j in range(len(v)):
                    reports.append('{} = {:.4f}'.format(k_list[j],v[j]))
            else:
                self.history[k].append(np.mean(v))
                reports.append('{} = {:.4f}'.format(k, np.mean(v)))
        #print('reports', reports)
        return ', '.join(reports)

    def log(self, n_classes):
        k_list=[]
        v_list=[]
        for k, v in sorted(self.history.items()):
            
            if k == "iou_class" or k == "dice_class":
                v_new = np.reshape(v, (-1, n_classes))
                v_means = v_new.mean(0)
                for i in range(len(v_means)):
                    k_list.append(k+str(i))
                    v_list.append(v_means[i])
            else:
                k_list.append(k)
                v_list.append(np.mean(v))

        #print("k", k_list, "v", v_list)

        metrics = {}
        for i in range(len(k_list)):
            metrics[k_list[i]] = v_list[i]

        #print('metrics',metrics)

        # metrics = {
        #     k: (sum(v) / len(v) if v else 0)
        #     for k, v in sorted(self.history.items())
        #     if k.startswith('val_') != self.is_train
        # }

        return ', '.join('average {} = {:.4f}'.format(name, value)
                         for name, value in metrics.items()).capitalize()

    def save(self, n_classes):
        """Save averaged metrics in this epoch to csv file."""

        if self.save_path is None:
            raise RuntimeError(
                'cannot save history without setting save_path.')
        k_list=[]
        v_list=[]
        for k, v in sorted(self.history.items()):
            #print("k",k,"v",v)
            # k_list.append(k)
            # v_list.append(np.mean(v))
            #print("k_list",k_list,"v_list",v_list)
            if k == "iou_class" or k == "dice_class":
                v_new = np.reshape(v, (-1, n_classes))
                v_means = v_new.mean(0)
                for i in range(len(v_means)):
                    k_list.append(k+str(i))
                    v_list.append(v_means[i])
            else:
                k_list.append(k)
                v_list.append(np.mean(v))

        # print(v_list)
        # print(k_list)
        keys =  k_list  #[k for k, _ in sorted(self.history.items())] #k_list 
        metrics =  v_list #[sum(v) / len(v) for _, v in sorted(self.history.items())] #v_list
        #print(metrics)
        if not os.path.exists(self.save_path):
            # create a new csv file
            with open(self.save_path, 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(keys + ['lr'])
                writer.writerow(metrics + [self.learning_rate])
        else:
            with open(self.save_path, 'a') as fp:
                writer = csv.writer(fp)
                writer.writerow(metrics + [self.learning_rate])

    def report(self, last_n_epochs=5):
        """Report training history summary.

        Arguments:
            last_n_epochs: number of final epochs to compute average losses and metrics.
        """

        df = pd.read_csv(self.save_path)

        metrics = '\n'.join(
            f'{key:20s} {df[key][-last_n_epochs:].mean():.4f}'
            for key in df.keys() if key not in ['lr', 'loss', 'val_loss']
        )

        return underline(
            '\nTraining Summary (Avg over last 5 epochs)', style='=') + '\n' + metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to history csv file')
    parser.add_argument('-l', '--last-n-epochs', type=int, default=5,
                        help='Number of final epochs to compute average losses and metrics')
    args = parser.parse_args()

    tracker = HistoryTracker(args.csv_path)
    tracker.report(args.last_n_epochs)
