import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# from torch_geometric.nn import knn_graph
from torch_cluster import knn_graph

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.8 ** ((epoch - 1) // 1))}

        # lr_adjust = {
        #     3: 5e-4, 6: 1e-4, 8: 5e-5,
        #     10: 1e-5, 15: 5e-6, 20: 1e-6
        # }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def plot_loss_curve(train_loss, val_loss, test_loss, save_path):
    """
    Plot the loss curve
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.plot(test_loss, label='Test Loss', color='seagreen')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_target_vs_pred(pred, true, save_path=None):
    assert pred.shape == true.shape, "Prediction and target shapes do not match"

    plt.figure(figsize=(10, 300))
    
    num_samples, num_pred, num_nodes = pred.shape

    for k in range(num_nodes):
        plt.subplot(num_nodes, 1, k + 1)
        for j in range(num_samples):
            plt.plot(range(1 + j, num_pred + 1 + j), pred[j,:,k], c='b', label='Prediction')  
            plt.plot(range(1 + j, num_pred + 1 + j), true[j,:,k], c='r', label='Target') 

    plt.title('Test prediction vs Target')
    plt.savefig(save_path)

def get_edge_indices(comm_centroid_file_path, k=5):
    df = pd.read_csv(comm_centroid_file_path)
    df.sort_values('area_id', inplace=True)

    features = ['centroid_x', 'centroid_y'] # except the area_id
    x = df[features].to_numpy() 
    x = torch.tensor(x, dtype=torch.float)

    # Get edge_index (2, num_edges)
    edge_index = knn_graph(x, k=k, loop=True)
    return edge_index