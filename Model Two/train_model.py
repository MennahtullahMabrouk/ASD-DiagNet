# Import libraries
import pandas as pd
import numpy as np
import numpy.ma as ma
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import functools
import time
import random

# Set parameters
p_ROI = "cc200"  # Options: cc200, dosenbach160, aal
p_fold = 10
p_center = "Stanford"
p_mode = "whole"
p_augmentation = True
p_Method = "ASD-DiagNet"

# Print parameters
print("*****List of parameters****")
print("ROI atlas: ", p_ROI)
print("Per Center or Whole: ", p_mode)
if p_mode == 'percenter':
    print("Center's name: ", p_center)
print("Method's name: ", p_Method)
if p_Method == "ASD-DiagNet":
    print("Augmentation: ", p_augmentation)

# Helper functions
def get_key(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3])
    else:
        key = '_'.join(f_split[0:2])
    return key

def get_label(filename):
    assert filename in labels
    return labels[filename]

def get_corr_data(filename):
    for file in os.listdir(data_main_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(data_main_path, file), sep='\t')
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()

def get_corr_matrix(filename):
    for file in os.listdir(data_main_path):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(data_main_path, file), sep='\t')
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        return corr

def confusion(g_truth, predictions):
    tn, fp, fn, tp = confusion_matrix(g_truth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def get_regs(samples_names, reg_num):
    datas = [all_corr[sn][0] for sn in samples_names]
    datas = np.array(datas)
    avg = np.mean(datas, axis=0)
    highs = avg.argsort()[-reg_num:][::-1]
    lows = avg.argsort()[:reg_num][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions

def norm_weights(samples_list):
    # Example: Return equal weights for all samples
    return np.ones(len(samples_list)) / len(samples_list)

def cal_similarity(eigvecs1, eigvecs2, weights, lim=2):
    # Example: Calculate cosine similarity between eigenvectors
    similarity = np.dot(eigvecs1[:lim], eigvecs2[:lim].T)
    return np.sum(similarity * weights)

def get_loader(data, samples_list, batch_size, mode='train', augmentation=False, aug_factor=1, num_neighbs=5, eig_data=None, similarity_fn=None, regions=None):
    dataset = CC200Dataset(data, samples_list, augmentation=augmentation, aug_factor=aug_factor, num_neighbs=num_neighbs, eig_data=eig_data, similarity_fn=similarity_fn, regions=regions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return loader

# Load data
data_main_path = '/home/taban/autism/paper_autism/acerta-abide/acerta-abide/data/functionals/cpac/filt_global/rois_' + p_ROI
flist = os.listdir(data_main_path)
flist = [get_key(f) for f in flist]

df_labels = pd.read_csv('/home/taban/autism/paper_autism/acerta-abide/acerta-abide/data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv')
df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})

labels = {row[1]['FILE_ID']: row[1]['DX_GROUP'] for row in df_labels.iterrows() if row[1]['FILE_ID'] != 'no_filename'}

# Compute correlations
if not os.path.exists('./correlations_file' + p_ROI + '.pkl'):
    all_corr = {f: (get_corr_data(f), get_label(f)) for f in flist}
    pickle.dump(all_corr, open('./correlations_file' + p_ROI + '.pkl', 'wb'))
else:
    all_corr = pickle.load(open('./correlations_file' + p_ROI + '.pkl', 'rb'))

# Compute eigenvalues and eigenvectors (if using ASD-DiagNet)
if p_Method == "ASD-DiagNet":
    eig_data = {}
    for f in flist:
        d = get_corr_matrix(f)
        eig_vals, eig_vecs = np.linalg.eig(d)
        sum_eigvals = np.sum(np.abs(eig_vals))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i], np.abs(eig_vals[i]) / sum_eigvals) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_data[f] = {'eigvals': np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals': np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs': [ep[1] for ep in eig_pairs]}

# Dataset class
class CC200Dataset(Dataset):
    def __init__(self, data, samples_list, augmentation=False, aug_factor=1, num_neighbs=5, eig_data=None, similarity_fn=None, regions=None):
        self.data = data
        self.flist = samples_list
        self.labels = np.array([self.data[f][1] for f in self.flist])
        self.augmentation = augmentation
        self.num_data = aug_factor * len(self.flist) if augmentation else len(self.flist)
        self.neighbors = {}
        if augmentation:
            weights = norm_weights(samples_list)
            for f in self.flist:
                label = self.data[f][1]
                candidates = set(np.array(self.flist)[self.labels == label]) - {f}
                eig_f = eig_data[f]['eigvecs']
                sim_list = [(similarity_fn(eig_f, eig_data[cand]['eigvecs'], weights), cand) for cand in candidates]
                sim_list.sort(key=lambda x: x[0], reverse=True)
                self.neighbors[f] = [item[1] for item in sim_list[:num_neighbs]]
        self.regions = regions

    def __getitem__(self, index):
        if index < len(self.flist):
            fname = self.flist[index]
            data = self.data[fname][0][self.regions]
            label = self.labels[index]
            return torch.FloatTensor(data), torch.FloatTensor([label])
        else:
            f1 = self.flist[index % len(self.flist)]
            d1, y1 = self.data[f1][0][self.regions], self.data[f1][1]
            f2 = np.random.choice(self.neighbors[f1])
            d2, y2 = self.data[f2][0][self.regions], self.data[f2][1]
            r = np.random.uniform(low=0, high=1)
            data = r * d1 + (1 - r) * d2
            return torch.FloatTensor(data), torch.FloatTensor([y1])

    def __len__(self):
        return self.num_data

# Autoencoder model
class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_latent, tied=True, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(num_latent, 1)
        )

    def forward(self, x, eval_classifier=False):
        x = torch.tanh(self.fc_encoder(x))
        x_logit = self.classifier(x) if eval_classifier else None
        x_rec = F.linear(x, self.fc_encoder.weight.t()) if self.tied else self.fc_decoder(x)
        return x_rec, x_logit

# Training and testing functions
def train(model, epoch, train_loader, p_bernoulli=None, mode='both', lam_factor=1.0):
    model.train()
    train_losses = []
    for batch_x, batch_y in train_loader:
        data, target = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        rec, logits = model(data, mode != 'ae')
        loss_ae = criterion_ae(rec, data) / len(batch_x) if mode in ['both', 'ae'] else 0
        loss_clf = criterion_clf(logits, target) if mode in ['both', 'clf'] else 0
        loss_total = loss_ae + lam_factor * loss_clf
        loss_total.backward()
        optimizer.step()
        train_losses.append([loss_ae.item(), loss_clf.item()])
    return train_losses

def test(model, criterion, test_loader, eval_classifier=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            data = batch_x.to(device)
            rec, logits = model(data, eval_classifier)
            if eval_classifier:
                proba = torch.sigmoid(logits).cpu().numpy()
                preds = (proba >= 0.5).astype(int)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds)
    return confusion(y_true, y_pred)

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if p_Method == "ASD-DiagNet":
    num_corr = len(all_corr[flist[0]][0])
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001
    num_epochs = 25
    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)
    crossval_res_kol = []

    for rp in range(10):
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        for train_index, test_index in kf.split(flist, y_arr):
            train_samples, test_samples = flist[train_index], flist[test_index]
            regions_inds = get_regs(train_samples, int(num_corr / 4))
            num_inpp = len(regions_inds)
            n_lat = int(num_inpp / 2)

            train_loader = get_loader(data=all_corr, samples_list=train_samples, batch_size=batch_size, mode='train',
                                      augmentation=p_augmentation, aug_factor=aug_factor, num_neighbs=num_neighbs,
                                      eig_data=eig_data, similarity_fn=sim_function, regions=regions_inds)
            test_loader = get_loader(data=all_corr, samples_list=test_samples, batch_size=batch_size, mode='test',
                                     augmentation=False, regions=regions_inds)

            model = MTAutoEncoder(num_inputs=num_inpp, num_latent=n_lat, tied=True, use_dropout=False).to(device)
            criterion_ae = nn.MSELoss(reduction='sum')
            criterion_clf = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD([{'params': model.fc_encoder.parameters(), 'lr': learning_rate_ae},
                                   {'params': model.classifier.parameters(), 'lr': learning_rate_clf}], momentum=0.9)

            for epoch in range(1, num_epochs + 1):
                train_losses = train(model, epoch, train_loader, mode='both' if epoch <= 20 else 'clf')

            res_mlp = test(model, criterion_ae, test_loader, eval_classifier=True)
            crossval_res_kol.append(res_mlp)

    print("Average results:", np.mean(np.array(crossval_res_kol), axis=0))