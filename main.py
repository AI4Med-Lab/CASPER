#JSG
import random
import warnings
import time
import h5py
import csv
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from datetime import datetime
import logging
import argparse
import json
import pandas as pd
import scipy.sparse as sp
from scipy import sparse
from scipy.stats import pearsonr
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split, KFold
from utils.compute_metrics import compute_metrics, spearmanrr
from utils.setup_logger import setup_logging
from models.CasperModel import Casper
from utils.centroidCalc import compute_centroids_leiden

torch.manual_seed(10)



print("Jai shri Ganesh!!!")
parser = argparse.ArgumentParser(description="Script that uses a JSON configuration file")
parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')

args = parser.parse_args()

try:
    with open(args.config, 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found at {args.config}")
except json.JSONDecodeError:
    print(f"Error: Failed to parse JSON file at {args.config}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_dir = os.path.join(params['result_path'], f"result_{timestamp}")
os.makedirs(model_save_dir,  exist_ok=True)

json_file = os.path.join(model_save_dir,f"experiment_info.json")

with open(json_file, "w") as exp:
    json.dump(params, exp, indent=4)


log_file = os.path.join(model_save_dir, "results_log.txt")
logger = setup_logging(log_file)
logger.info("Logging setup complete.")
logger.info(f"Experiment information saved to the path: {json_file}")


metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")

csv_dir = os.path.join(model_save_dir, "fold_results")
os.makedirs(csv_dir,  exist_ok=True)

n_splits = 5 
adata_spatial = sc.read_h5ad(f"{params['data_path']}/{params['datasert_name']}/spatial.h5ad")
adata_seq = sc.read_h5ad(f"{params['data_path']}/{params['datasert_name']}/scRNA_common.h5ad")


sc.pp.filter_cells(adata_seq, min_counts=1)
sc.pp.filter_genes(adata_seq, min_counts=1)
sc.pp.filter_cells(adata_spatial, min_counts=1)
sc.pp.filter_genes(adata_spatial, min_counts=1)

sc.pp.normalize_total(adata_seq, target_sum=1e4)
sc.pp.log1p(adata_seq)

sc.pp.normalize_total(adata_spatial, target_sum=1e4)
sc.pp.log1p(adata_spatial)



# Write CSV headers (only once before the loop starts)
with open(metrics_csv_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'fold','epoch','phase','loss','pearson_mean', 'spearman_mean_genewise',
        'l1_error_mean','l2_errors_mean','r2_scores_mean',
        'pearson_std','l2_error_q1','l2_error_q2','l2_error_q3',
        'r2_score_q1','r2_score_q2','r2_score_q3'
    ])
    writer.writeheader()



#+++++++++++++++++++++++++++++++++++++++++ Just Added ++++++++++++++++++++++++++++++++++++++++++
def run_5fold_training(st_adata, sc_adata, common_genes, params):
    """
    st_adata: AnnData with only the common genes (spots x genes)
    sc_adata: AnnData with only the common genes (cells x genes)
    """

    def get_device(): return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if "datasert_name" in params:
        logger.info(f"[info] Dataset name provided: {params['datasert_name']}") 
    sp_idx_train = np.load(f"{params['idx_path']}/{params['datasert_name']}_train_indices.npy")
    sp_idx_test  = np.load(f"{params['idx_path']}/{params['datasert_name']}_test_indices.npy")




    print(f"The test indices are ::::: {sp_idx_test}")

    st_adata_test = st_adata[sp_idx_test].copy()
    st_adata = st_adata[sp_idx_train].copy()

    logger.info(f"[info] Spatial data split: train={st_adata.n_obs}, test={st_adata_test.n_obs}")

    device = get_device()
    logger.info(f"[info] Using device: {device}")
    raw_shared_gene = np.array(common_genes)
    # Optional HVG selection parameters
    n_top_hvgs = params.get("n_top_hvg", None)
    use_hvg = params.get("use_hvg", False)
    logger.info(f"[info] use_hvg={use_hvg}, n_top_hvgs={n_top_hvgs}")

    # 1) Compute centroids from scRNA (Leiden-based)
    if params['datasert_name']=='osmFISH':
        n_pcs = 13
    else:
        n_pcs = params.get("n_pcs", 50)

    print(f"++++++++++++++++++++ sc_adata {sc_adata.shape}")
    centroids_df, ad_sc_sub = compute_centroids_leiden(sc_adata,
                                                       logger,
                                                       n_top_hvgs=n_top_hvgs if use_hvg else None,
                                                       n_pcs=n_pcs,
                                                       neighbors_k=params.get("neighbors_k", 15),
                                                       leiden_resolution=params.get("leiden_resolution", 1.0),
                                                       use_hvg_subset=use_hvg)
    # centroids_df rows ~= clusters, cols = genes (order corresponds to st_adata.var_names because we subset earlier)
    centroids = centroids_df.values.astype(np.float32)  # shape: (n_centroids, n_genes)
    n_centroids = centroids.shape[0]

    print(f"n_centroids ", n_centroids)

    # data matrices (numpy)
    st_X = st_adata.X.copy()
    if sp.issparse(st_X):
        st_X = st_X.toarray()
    sc_X = sc_adata.X.copy()
    if sp.issparse(sc_X):
        sc_X = sc_X.toarray()

    st_X_test = st_adata_test.X.toarray() if sp.issparse(st_adata_test.X) else st_adata_test.X

    n_spots, n_genes = st_X.shape
    logger.info(f"[info] After intersection: n_spots={n_spots}, n_genes={n_genes}, n_centroids={n_centroids}")

    # 2) 5-fold gene-wise CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=params.get("rand_seed", 0))
    all_pred = np.zeros_like(st_X_test, dtype=np.float32)
    fold_metrics = {}

    class SpotDataset(Dataset):
        def __init__(self, X_in: np.ndarray, Y_target: np.ndarray):
            assert X_in.shape[0] == Y_target.shape[0]
            self.X = torch.from_numpy(X_in.astype(np.float32))
            self.Y = torch.from_numpy(Y_target.astype(np.float32))
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]


    for fold, (train_idx, test_idx) in enumerate(kf.split(raw_shared_gene), start=1):
        
        best_spearman = -np.inf
        best_state = None
        wait = 0

        train_gene = list(raw_shared_gene[train_idx])
        test_gene = list(raw_shared_gene[test_idx])
        logger.info(f"\n[fold {fold}] Train genes: {len(train_idx)} | Test genes: {len(test_idx)}")

        # Split input and target genes
        X_in = st_X[:, train_idx]        # (n_spots, n_input_genes)
        Y_target = st_X[:, test_idx]     # ground truth (n_spots, n_test_genes)

        X_in_test = st_X_test[:, train_idx]
        Y_target_test = st_X_test[:, test_idx]

        sc_centroid_tensor = torch.from_numpy(centroids).to(device)  # (n_centroids, n_genes)
        print("sc_centroids shape:", sc_centroid_tensor.shape)

        print("X_in shape:", X_in.shape)
        print("sc_X shape:", sc_X.shape)
        # Build model
        st_in_dim = X_in.shape[1]
        sc_gene_dim = sc_centroid_tensor.shape[1]
        emb_dim = 256
        out_dim = len(test_idx)
        model = Casper(
            st_dim=st_in_dim,
            sc_dim=sc_gene_dim,
            out_dim=out_dim,
            emb_dim=emb_dim,
            n_heads=params.get("n_heads", 4),
            dropout=params.get("dropout", 0.1)
        ).to(device)

        if fold == 1:
            logger.info(f"Model architecture:\n{model}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params.get("lr", 1e-3),
                                     weight_decay=params.get("weight_decay", 1e-5))
        loss_fn = nn.MSELoss()

        # Dataset + DataLoader
        dataset_train = SpotDataset(X_in, Y_target)

        dataset_test = SpotDataset(X_in_test, Y_target_test)

        train_loader = DataLoader(dataset_train, batch_size=params.get("batch_size", 128),
                            shuffle=True)
        test_loader =  DataLoader(dataset_test, batch_size=params.get("batch_size", 128),
                            shuffle=False)

        # Early stopping setup
        best_val = np.inf
        best_state = None
        patience = params.get("early_stop_patience", 5)
        wait = 0
        epochs = params.get("epochs", 100)

        # =========================
        #   EPOCH LOOP
        # =========================
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()

            epoch_train_loss = 0.0
            all_y_true, all_y_pred = [], []
            pearson_train_corr, spearman_train_corr = [], []

            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                preds, _ = model(xb, sc_centroid_tensor)
                loss = loss_fn(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ---- Metrics ----
                try:
                    batch_pearson, _ = pearsonr(
                        preds.detach().cpu().numpy().flatten(),
                        yb.detach().cpu().numpy().flatten()
                    )
                except Exception:
                    batch_pearson = np.nan

                pearson_train_corr.append(batch_pearson)

                batch_spearman = spearmanrr(
                    preds.detach().cpu(), yb.detach().cpu()
                )
                spearman_train_corr.append(batch_spearman)

                epoch_train_loss += loss.item()
                all_y_true.append(yb.detach().cpu().numpy())
                all_y_pred.append(preds.detach().cpu().numpy())

            # ====== End of training for this epoch ======
            avg_train_loss = epoch_train_loss / len(train_loader)
            all_y_true = np.vstack(all_y_true)
            all_y_pred = np.vstack(all_y_pred)
            results_train = compute_metrics(all_y_true, all_y_pred, genes=None)
            results_train.update({
                'fold': fold,
                'epoch': ep,
                'phase': 'train',
                'loss': round(avg_train_loss, 4)
            })

            logger.info(f"[Fold {fold}] Epoch {ep} Train Loss={avg_train_loss:.4f} | "
                        f"Pearson={np.nanmean(pearson_train_corr):.4f} | "
                        f"Spearman={np.nanmean(spearman_train_corr):.4f} | "
                        f"Time={time.time()-t0:.1f}s")

            # ========== VALIDATION ==========

            preds_all, Y_val_all = [], []


            model.eval()
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds, _ = model(xb, sc_centroid_tensor)
                    preds_all.append(preds.cpu().numpy())
                    Y_val_all.append(yb.cpu().numpy())

            preds_all = np.vstack(preds_all)
            Y_val_all = np.vstack(Y_val_all)

            val_loss = mean_squared_error(Y_val_all.reshape(-1), preds_all.reshape(-1))
            results_val = compute_metrics(Y_val_all, preds_all)
            results_val.update({
                'fold': fold,
                'epoch': ep,
                'phase': 'val',
                'loss': round(val_loss, 4)
            })
            print(f"results_val: {results_val}")

            logger.info(f"[Fold {fold}] Epoch {ep} Val Loss={val_loss:.4f} | "
                        f"Pearson={results_val['pearson_mean']:.4f} | "
                        f"Spearman={results_val['spearman_mean_genewise']:.4f} | "
                        f"Time={time.time()-t0:.1f}s")

            # ========== WRITE BOTH TRAIN AND VAL ROWS ==========
            with open(metrics_csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'fold','epoch','phase','loss','pearson_mean', 'spearman_mean_genewise',
                    'l1_error_mean','l2_errors_mean','r2_scores_mean',
                    'pearson_std','l2_error_q1','l2_error_q2','l2_error_q3',
                    'r2_score_q1','r2_score_q2','r2_score_q3'
                ])
                writer.writerow({key: results_train.get(key, "") for key in writer.fieldnames})
                writer.writerow({key: results_val.get(key, "") for key in writer.fieldnames})

            # ========== EARLY STOPPING ==========
            if float(results_val['spearman_mean_genewise']) > float(best_spearman):

                #results save ======= 
                all_pred[:, test_idx] = preds_all
                df_pred = pd.DataFrame(preds_all, columns=test_gene)
                df_true = pd.DataFrame(Y_target_test, columns=test_gene)
                df_pred.to_csv(os.path.join(csv_dir, f"fold_{fold}_pred.csv"), index=False)
                df_true.to_csv(os.path.join(csv_dir, f"fold_{fold}_true.csv"), index=False)


                best_spearman = results_val['spearman_mean_genewise']
                best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"[Fold {fold}] Early stopping at epoch {ep}. Best Spearman={best_spearman:.4f}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state['model'])
            logger.info(f"[Fold {fold}] Loaded best model (Spearman={best_spearman:.4f}) before saving.")
        else:
            logger.warning(f"[Fold {fold}] No improvement recorded — saving final model instead.")
            best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}

        # After training, save the best model for this fold
        ckpt_path = os.path.join(model_save_dir, f"model_fold{fold}.pt")
        torch.save(best_state['model'], ckpt_path)
        logger.info(f"[Fold {fold}] Saved best model checkpoint: {ckpt_path}")
    return all_pred
#+++++++++++++++++++++++++++++++++++++++++ Just Added ++++++++++++++++++++++++++++++++++++++++++


common_genes = [g for g in adata_seq.var_names if g in adata_spatial.var_names]
logger.info(f"[info] Found {len(common_genes)} common genes between scRNA and ST data.")

if len(common_genes) == 0:
    logger.error("❌ No overlapping genes found between scRNA and spatial datasets.")
    sys.exit(1)

adata_seq = adata_seq[:, common_genes].copy()
adata_spatial = adata_spatial[:, common_genes].copy()

logger.info(f"[info] Subsetting done: scRNA ({adata_seq.shape}) | ST ({adata_spatial.shape})")

# Run training and imputation
logger.info("[info] Starting 5-fold gene-wise cross-validation training...")

all_pred_matrix = run_5fold_training(adata_spatial, adata_seq, common_genes, params)
pd.DataFrame(all_pred_matrix, columns=common_genes).to_csv(
    os.path.join(csv_dir, "attention_imputes.csv"), index=True
)


