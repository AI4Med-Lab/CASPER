
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from typing import Optional
import logging

def compute_centroids_leiden(
    adata_sc,
    logger,
    n_top_hvgs: Optional[int] = 2000,
    n_pcs: int = 30,
    neighbors_k: int = 15,
    leiden_resolution: float = 1.0,
    use_hvg_subset: bool = True,
):
    """
    Compute cluster centroids from full scRNA-seq AnnData using:
      (Optional) HVG -> PCA -> Neighbors -> Leiden.
    Returns:
      centroids_df (pd.DataFrame): rows = clusters, cols = genes
      adata_proc (AnnData): processed AnnData (possibly HVG subset)
    """
    logger.info("[info] Starting full scRNA preprocessing for centroid computation...")
    ad = adata_sc.copy()

    # -----------------------------
    # Step 1. HVG selection (optional)
    # -----------------------------
    if use_hvg_subset and n_top_hvgs and n_top_hvgs > 0:
        logger.info(f"[info] Selecting top {n_top_hvgs} highly variable genes.")

        # Heuristic: check if data is raw counts or log-transformed
        X = ad.X.toarray() if sp.issparse(ad.X) else ad.X
        if np.max(X) < 50 or not np.allclose(X, np.round(X)):
            # Log-transformed or normalized
            hv_flavor = "cell_ranger"
            logger.info("[warn] Non-integer or normalized values detected — using flavor='cell_ranger'.")
        else:
            # Raw count data
            hv_flavor = "seurat_v3"
            logger.info("[info] Integer count data detected — using flavor='seurat_v3'.")

        # Run HVG selection
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_hvgs, flavor=hv_flavor)
        ad = ad[:, ad.var["highly_variable"]].copy()
        logger.info(f"[info] Retained {ad.n_vars} HVGs for PCA and clustering.")
    else:
        logger.info("[info] Using full gene set (no HVG subset).")

    # -----------------------------
    # Step 2. Handle sparse & NaN data
    # -----------------------------
    if sp.issparse(ad.X):
        ad.X = ad.X.toarray()
    ad.X = np.nan_to_num(ad.X, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(ad.X).any():
        logger.info("[warn] NaN values still detected after cleanup - replacing with 0.")
        ad.X = np.nan_to_num(ad.X, nan=0.0)

    # -----------------------------
    # Step 3. PCA + neighbors + Leiden
    # -----------------------------
    logger.info("[info] Running PCA on cleaned data.")
    sc.pp.pca(ad, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=neighbors_k, n_pcs=n_pcs)
    sc.tl.leiden(
        ad,
        resolution=leiden_resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    n_clusters = len(ad.obs["leiden"].unique())
    logger.info(f"[info] Leiden clustering complete - found {n_clusters} clusters.")

    # -----------------------------
    # Step 4. Compute centroids
    # -----------------------------
    logger.info("[info] Computing cluster centroids.")
    X = ad.X.toarray() if sp.issparse(ad.X) else ad.X
    expr_df = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
    cluster_series = ad.obs["leiden"].astype(str)
    centroids_df = expr_df.groupby(cluster_series).mean()

    logger.info(f"[info] Computed {centroids_df.shape[0]} centroids vs {centroids_df.shape[1]} genes.")
    logger.info("[info] Centroid computation finished successfully.")
    return centroids_df, ad



def compute_cluster_random_samples_leiden(
    adata_sc,
    logger,
    n_top_hvgs: Optional[int] = 2000,
    n_pcs: int = 30,
    neighbors_k: int = 15,
    leiden_resolution: float = 1.0,
    use_hvg_subset: bool = True,
    n_samples_per_cluster: int = 10,
):
    """
    Perform Leiden clustering on full scRNA-seq AnnData and 
    randomly sample cells from each cluster (instead of centroids).

    Args:
      adata_sc (AnnData): input single-cell dataset
      logger: logging object
      n_top_hvgs (int): number of HVGs to select (if use_hvg_subset=True)
      n_pcs (int): number of PCs for PCA
      neighbors_k (int): K for nearest neighbor graph
      leiden_resolution (float): resolution for Leiden clustering
      use_hvg_subset (bool): whether to use only HVGs
      n_samples_per_cluster (int): number of random cells to sample per cluster

    Returns:
      sampled_df (pd.DataFrame): expression matrix of sampled cells (rows=cells, cols=genes)
      adata_proc (AnnData): processed AnnData with Leiden clusters
    """
    logger.info("[info] Starting full scRNA preprocessing for cluster-based sampling...")
    ad = adata_sc.copy()

    # -----------------------------
    # Step 1. HVG selection (optional)
    # -----------------------------
    if use_hvg_subset and n_top_hvgs and n_top_hvgs > 0:
        logger.info(f"[info] Selecting top {n_top_hvgs} highly variable genes.")
        X = ad.X.toarray() if sp.issparse(ad.X) else ad.X
        if np.max(X) < 50 or not np.allclose(X, np.round(X)):
            hv_flavor = "cell_ranger"
            logger.info("[warn] Non-integer or normalized values detected — using flavor='cell_ranger'.")
        else:
            hv_flavor = "seurat_v3"
            logger.info("[info] Integer count data detected — using flavor='seurat_v3'.")

        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_hvgs, flavor=hv_flavor)
        ad = ad[:, ad.var["highly_variable"]].copy()
        logger.info(f"[info] Retained {ad.n_vars} HVGs for PCA and clustering.")
    else:
        logger.info("[info] Using full gene set (no HVG subset).")

    # -----------------------------
    # Step 2. Handle sparse & NaN data
    # -----------------------------
    if sp.issparse(ad.X):
        ad.X = ad.X.toarray()
    ad.X = np.nan_to_num(ad.X, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(ad.X).any():
        logger.info("[warn] NaN values still detected after cleanup - replacing with 0.")
        ad.X = np.nan_to_num(ad.X, nan=0.0)

    # -----------------------------
    # Step 3. PCA + neighbors + Leiden
    # -----------------------------
    logger.info("[info] Running PCA + Leiden clustering.")
    sc.pp.pca(ad, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=neighbors_k, n_pcs=n_pcs)
    sc.tl.leiden(
        ad,
        resolution=leiden_resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    n_clusters = len(ad.obs["leiden"].unique())
    logger.info(f"[info] Leiden clustering complete — found {n_clusters} clusters.")

    # -----------------------------
    # Step 4. Random sampling per cluster
    # -----------------------------
    logger.info(f"[info] Sampling up to {n_samples_per_cluster} random cells per cluster.")

    X = ad.X.toarray() if sp.issparse(ad.X) else ad.X
    expr_df = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
    cluster_series = ad.obs["leiden"].astype(str)

    sampled_indices = []
    for cluster_id, indices in cluster_series.groupby(cluster_series).groups.items():
        cluster_cells = list(indices)
        n_cluster_cells = len(cluster_cells)
        n_to_sample = min(n_samples_per_cluster, n_cluster_cells)
        sampled_cells = np.random.choice(cluster_cells, size=n_to_sample, replace=False)
        sampled_indices.extend(sampled_cells)
        logger.info(f"[info] Cluster {cluster_id}: sampled {n_to_sample}/{n_cluster_cells} cells.")

    sampled_df = expr_df.loc[sampled_indices]
    logger.info(f"[info] Final sampled matrix shape: {sampled_df.shape}")

    logger.info("[info] Random sampling from clusters completed successfully.")
    return sampled_df, ad
