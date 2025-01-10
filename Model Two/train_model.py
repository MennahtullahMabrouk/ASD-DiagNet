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
import functools
import random
import gc  # For garbage collection
import logging  # For logging messages

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set parameters
p_ROI = "aal"  # Options: aal, cc200, dosenbach160
p_fold = 10
p_center = "Stanford"
p_mode = "whole"
p_augmentation = True
p_Method = "ASD-DiagNet"

# Print parameters
logger.info("*****List of parameters****")
logger.info(f"ROI atlas: {p_ROI}")
logger.info(f"Per Center or Whole: {p_mode}")
if p_mode == 'percenter':
    logger.info(f"Center's name: {p_center}")
logger.info(f"Method's name: {p_Method}")
if p_Method == "ASD-DiagNet":
    logger.info(f"Augmentation: {p_augmentation}")

# Helper functions
def get_key(filename):
    return filename.replace('_rois_aal.1D', '')

def get_label(filename, labels):
    assert filename in labels, f"Filename {filename} not found in labels."
    return labels[filename]

def get_corr_data(filepath):
    logger.info(f"Processing file: {filepath}")
    # Load the .1D file
    data = np.loadtxt(filepath)
    if data.ndim == 1:  # If the data is 1D, reshape it to (timepoints, 1)
        data = data.reshape(-1, 1)
    elif data.ndim == 2:  # If the data is 2D, ensure it's (timepoints, regions)
        if data.shape[0] < 2:  # Check if there are at least 2 timepoints
            logger.warning(f"Skipping {filepath}: Not enough timepoints for correlation matrix calculation.")
            return None
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Compute the correlation matrix
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(data.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()

def get_corr_matrix(filepath):
    logger.info(f"Processing file: {filepath}")
    # Load the .1D file
    data = np.loadtxt(filepath)
    if data.ndim == 1:  # If the data is 1D, reshape it to (timepoints, 1)
        data = data.reshape(-1, 1)
    elif data.ndim == 2:  # If the data is 2D, ensure it's (timepoints, regions)
        if data.shape[0] < 2:  # Check if there are at least 2 timepoints
            logger.warning(f"Skipping {filepath}: Not enough timepoints for correlation matrix calculation.")
            return None
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Compute the correlation matrix
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(data.T))
        return corr

def confusion(g_truth, predictions):
    tn, fp, fn, tp = confusion_matrix(g_truth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def get_regs(samples_names, reg_num):
    logger.info(f"Selecting regions for {len(samples_names)} samples")
    datas = [all_corr[sn][0] for sn in samples_names]
    datas = np.array(datas)
    avg = np.mean(datas, axis=0)
    highs = avg.argsort()[-reg_num:][::-1]
    lows = avg.argsort()[:reg_num][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions

def norm_weights(samples_list):
    return np.ones(len(samples_list))  # Return weights as a 1D array of ones

def cal_similarity(eigvecs1, eigvecs2, weights, lim=2):
    eigvecs1 = np.array(eigvecs1)  # Convert to NumPy array
    eigvecs2 = np.array(eigvecs2)  # Convert to NumPy array
    similarity = np.dot(eigvecs1[:lim], eigvecs2[:lim].T)
    # Use weights as a scalar (e.g., average weight)
    weights = np.mean(weights)  # Use the average weight
    return np.sum(similarity) * weights

def get_loader(data, samples_list, batch_size, mode='train', augmentation=False, aug_factor=1, num_neighbs=5, eig_data=None, similarity_fn=None, regions=None):
    logger.info(f"Creating DataLoader for {mode} mode with {len(samples_list)} samples")
    dataset = CC200Dataset(data, samples_list, augmentation=augmentation, aug_factor=aug_factor, num_neighbs=num_neighbs, eig_data=eig_data, similarity_fn=similarity_fn, regions=regions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return loader

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_main_path = os.path.join(base_dir, 'abide_data/cpac/nofilt_noglobal/rois_' + p_ROI)
phenotypic_csv_path = os.path.join(base_dir, 'abide_data/Phenotypic_V1_0b.csv')
correlations_file_path = os.path.join(base_dir, 'correlations_file_' + p_ROI + '.pkl')

# Validate paths
if not os.path.exists(data_main_path):
    raise FileNotFoundError(f"Data path does not exist: {data_main_path}")
logger.info(f"Data path exists: {data_main_path}")
logger.info(f"Files in directory: {os.listdir(data_main_path)}")

# Generate flist with all .1D files
flist = []
for root, dirs, files in os.walk(data_main_path):
    for file in files:
        if file.endswith('.1D'):
            key = get_key(file)
            flist.append(key)
logger.info(f"Filtered file list (flist): {flist}")

# Ensure flist is not empty
if not flist:
    raise FileNotFoundError(f"No valid files found in {data_main_path}. Please check the directory or ROI parameter.")

# Load phenotypic data
logger.info("Loading phenotypic data")
df_labels = pd.read_csv(phenotypic_csv_path)
df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})
labels = {row[1]['FILE_ID']: row[1]['DX_GROUP'] for row in df_labels.iterrows() if row[1]['FILE_ID'] != 'no_filename'}

# Load or compute correlations
if not os.path.exists(correlations_file_path):
    logger.info("Computing correlations from scratch")
    all_corr = {}
    for f in flist:
        filepath = os.path.join(data_main_path, f + '_rois_aal.1D')
        corr_data = get_corr_data(filepath)
        if corr_data is not None:  # Skip invalid data
            all_corr[f] = (corr_data, get_label(f, labels))
    with open(correlations_file_path, 'wb') as f:
        pickle.dump(all_corr, f)
else:
    logger.info("Loading precomputed correlations")
    with open(correlations_file_path, 'rb') as f:
        all_corr = pickle.load(f)

# Free up memory
del df_labels
gc.collect()

if p_Method == "ASD-DiagNet":
    logger.info("Starting ASD-DiagNet method")
    eig_data = {}
    batch_size = 10  # Process 10 files at a time
    for i in range(0, len(flist), batch_size):
        batch_files = flist[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of {len(flist) // batch_size + 1}")
        for f in batch_files:
            filepath = os.path.join(data_main_path, f + '_rois_aal.1D')
            d = get_corr_matrix(filepath)
            if d is not None and d.ndim == 2:  # Ensure the correlation matrix is 2D
                try:
                    eig_vals, eig_vecs = np.linalg.eig(d)
                    sum_eigvals = np.sum(np.abs(eig_vals))
                    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i], np.abs(eig_vals[i]) / sum_eigvals) for i in range(len(eig_vals))]
                    eig_pairs.sort(key=lambda x: x[0], reverse=True)
                    eig_data[f] = {'eigvals': np.array([ep[0] for ep in eig_pairs]),
                                   'norm-eigvals': np.array([ep[2] for ep in eig_pairs]),
                                   'eigvecs': [ep[1] for ep in eig_pairs]}
                except np.linalg.LinAlgError as e:
                    logger.error(f"Skipping {f}: Error in eigenvalue decomposition - {e}")
            else:
                logger.warning(f"Skipping {f}: Invalid correlation matrix shape - {d.shape if d is not None else 'None'}")
        # Free up memory after each batch
        del batch_files
        gc.collect()

class CC200Dataset(Dataset):
    def __init__(self, data, samples_list, augmentation=False, aug_factor=1, num_neighbs=5, eig_data=None, similarity_fn=None, regions=None):
        self.data = data
        self.flist = samples_list
        self.labels = np.array([self.data[f][1] for f in self.flist])
        self.augmentation = augmentation
        self.num_data = aug_factor * len(self.flist) if augmentation else len(self.flist)
        self.neighbors = {}
        if augmentation:
            logger.info(f"Applying data augmentation with factor {aug_factor}")
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

def train(model, epoch, train_loader, mode='both', lam_factor=1.0):
    model.train()
    train_losses = []  # Initialize as a list
    logger.info(f"Starting training for epoch {epoch} in mode {mode}")
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        data, target = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        rec, logits = model(data, mode != 'ae')
        loss_ae = criterion_ae(rec, data) / len(batch_x) if mode in ['both', 'ae'] else 0
        loss_clf = criterion_clf(logits, target) if mode in ['both', 'clf'] else 0
        loss_total = loss_ae + lam_factor * loss_clf
        loss_total.backward()
        optimizer.step()
        # Append losses as a list
        train_losses.append([loss_ae.item() if mode in ['both', 'ae'] else 0,
                             loss_clf.item() if mode in ['both', 'clf'] else 0])
        if batch_idx % 10 == 0:  # Log every 10 batches
            logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss AE = {loss_ae.item() if mode in ['both', 'ae'] else 0}, Loss CLF = {loss_clf.item() if mode in ['both', 'clf'] else 0}")
    return train_losses

def test(model, test_loader, eval_classifier=False):
    model.eval()
    y_true, y_pred = [], []
    logger.info("Starting testing")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            data = batch_x.to(device)
            _, logits = model(data, eval_classifier)
            if eval_classifier:
                proba = torch.sigmoid(logits).cpu().numpy()
                preds = (proba >= 0.5).astype(int)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds)
    accuracy, sensitivity, specificity = confusion(y_true, y_pred)
    logger.info(f"Test results: Accuracy = {accuracy}, Sensitivity = {sensitivity}, Specificity = {specificity}")
    return accuracy, sensitivity, specificity

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS backend on Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU if available
else:
    device = torch.device("cpu")  # Fallback to CPU
logger.info(f"Using device: {device}")

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

    # Shuffle the data once before splitting into folds
    np.random.shuffle(flist)
    y_arr = np.array([get_label(f, labels) for f in flist])

    # Initialize StratifiedKFold once
    kf = StratifiedKFold(n_splits=p_fold, random_state=42, shuffle=True)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(flist, y_arr)):
        logger.info(f"Starting fold {fold_idx + 1} of {p_fold}")
        logger.info(f"Train indices: {train_index}, Test indices: {test_index}")  # Verify uniqueness
        train_samples = [flist[i] for i in train_index]
        test_samples = [flist[i] for i in test_index]
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

        res_mlp = test(model, test_loader, eval_classifier=True)
        crossval_res_kol.append(res_mlp)
        logger.info(f"Fold {fold_idx + 1} results: Accuracy = {res_mlp[0]}, Sensitivity = {res_mlp[1]}, Specificity = {res_mlp[2]}")

    logger.info("Cross-validation completed")
    logger.info(f"Average results: Accuracy = {np.mean(np.array(crossval_res_kol), axis=0)[0]}, Sensitivity = {np.mean(np.array(crossval_res_kol), axis=0)[1]}, Specificity = {np.mean(np.array(crossval_res_kol), axis=0)[2]}")