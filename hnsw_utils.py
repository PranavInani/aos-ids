"""
HNSW-based Anomaly Detection Utilities for AOC-IDS.

Replaces the 1D Gaussian-fit decision boundary with a high-dimensional
Hierarchical Navigable Small World (HNSW) graph search over autoencoder
embeddings.  Instead of projecting embeddings onto a single cosine-similarity
axis and fitting two Gaussians, this module:

  1. Builds an HNSW index from *normal* training embeddings (encoder & decoder).
  2. For each test sample, queries the k nearest normal neighbours.
  3. Classifies as anomalous if the mean k-NN distance exceeds a threshold
     learned from the training set (percentile-based).

Advantages over Gaussian fit:
  - Captures complex, non-linear clusters of normal behaviour.
  - Works in the full embedding space (no information-lossy 1D projection).
  - O(log N) query time via HNSW vs. O(N) Gaussian MLE per evaluation.

Dependencies: hnswlib (pip install hnswlib)
"""

from utils import AE, CRCLoss, SplitData, load_data, setup_seed, score_detail
import torch
import torch.nn.functional as F
import numpy as np

try:
    import hnswlib
except ImportError:
    raise ImportError(
        "hnswlib is required for the HNSW anomaly detection module.\n"
        "Install it with:  pip install hnswlib"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  HNSW Index Builder
# ═══════════════════════════════════════════════════════════════════════════

class HNSWAnomalyDetector:
    """
    Builds an HNSW index from normal-class embeddings and classifies new
    samples as normal/anomalous based on k-NN distance.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to query.
    threshold_percentile : float
        Percentile of normal-to-normal k-NN distances used as the decision
        boundary.  E.g. 95 means only 5% of normal training samples exceed
        the threshold, so anything above is anomalous.
    ef_construction : int
        HNSW construction-time search width (higher = more accurate index,
        slower build).
    M : int
        HNSW max number of bi-directional links per node per layer.
    ef_search : int
        HNSW query-time search width (higher = more accurate query, slower).
    space : str
        Distance metric — 'cosine' or 'l2'.
    """

    def __init__(self, k=10, threshold_percentile=95.0,
                 ef_construction=200, M=16, ef_search=100, space='cosine'):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.space = space
        self.index = None
        self.threshold = None
        self._normal_distances = None   # cached for confidence calculation

    def build(self, normal_embeddings: np.ndarray):
        """
        Build the HNSW index from normal embeddings and compute the
        anomaly threshold from the training distances.

        Parameters
        ----------
        normal_embeddings : np.ndarray, shape (N, D)
            L2-normalised embeddings of normal training samples.
        """
        n, dim = normal_embeddings.shape
        k_actual = min(self.k + 1, n)  # +1 because query includes itself

        # Build HNSW index
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(max_elements=n, ef_construction=self.ef_construction, M=self.M)
        self.index.add_items(normal_embeddings, np.arange(n))
        self.index.set_ef(self.ef_search)

        # Self-query to learn the distance distribution of normal samples
        labels, distances = self.index.knn_query(normal_embeddings, k=k_actual)
        # Exclude self-match (distance ≈ 0) — first column
        if k_actual > 1:
            knn_distances = distances[:, 1:]  # shape (N, k)
        else:
            knn_distances = distances  # fallback when N is tiny
        mean_knn_dist = np.mean(knn_distances, axis=1)  # (N,)

        self._normal_distances = mean_knn_dist
        self.threshold = float(np.percentile(mean_knn_dist, self.threshold_percentile))

    def query(self, embeddings: np.ndarray):
        """
        Classify embeddings as normal (0) or anomalous (1).

        Returns
        -------
        predictions : np.ndarray[int32], shape (M,)
            0 = normal, 1 = anomalous.
        confidence : np.ndarray[float32], shape (M,)
            Normalised distance-based confidence in [0, 1].
            Higher = more confident about the prediction.
        mean_distances : np.ndarray[float32], shape (M,)
            Raw mean k-NN distances (useful for debugging).
        """
        assert self.index is not None, "Must call build() before query()"

        n = embeddings.shape[0]
        max_elements = self.index.get_max_elements()
        k_actual = min(self.k, max_elements)

        labels, distances = self.index.knn_query(embeddings, k=k_actual)
        mean_dist = np.mean(distances, axis=1).astype(np.float32)

        # Classification
        predictions = (mean_dist > self.threshold).astype(np.int32)

        # Confidence: how far from the threshold (normalised)
        # distance_ratio > 1 → anomalous, < 1 → normal
        distance_ratio = mean_dist / (self.threshold + 1e-8)
        # confidence = |1 - ratio|, clipped and normalised to [0, 1]
        raw_confidence = np.abs(1.0 - distance_ratio)
        p95 = np.percentile(raw_confidence, 95) if len(raw_confidence) > 0 else 1.0
        if p95 > 0:
            confidence = np.clip(raw_confidence / p95, 0.0, 1.0).astype(np.float32)
        else:
            confidence = np.ones_like(raw_confidence, dtype=np.float32)

        return predictions, confidence, mean_dist

    def add_items(self, new_embeddings: np.ndarray):
        """
        Incrementally add new normal embeddings to the existing index.
        This is the key advantage of HNSW for online learning — the index
        grows without full rebuild.
        """
        assert self.index is not None, "Must call build() before add_items()"
        current_max = self.index.get_max_elements()
        n_new = new_embeddings.shape[0]
        needed = self.index.get_current_count() + n_new
        if needed > current_max:
            self.index.resize_index(int(needed * 1.5))
        start_id = self.index.get_current_count()
        ids = np.arange(start_id, start_id + n_new)
        self.index.add_items(new_embeddings, ids)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate with HNSW (replaces Gaussian-based evaluate_with_confidence)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_embeddings(model, x, layer_idx):
    """Extract and L2-normalise embeddings from a model."""
    with torch.no_grad():
        out = model(x)[layer_idx]
    return F.normalize(out, p=2, dim=1).cpu().numpy()


def evaluate_hnsw(x_train, y_train, x_test, y_test, model,
                  k=10, threshold_percentile=95.0, ef_construction=200,
                  M=16, ef_search=100):
    """
    HNSW-based evaluation replacing the Gaussian fit.

    For BOTH encoder (layer 0) and decoder (layer 1) branches:
      1. Extract L2-normalised embeddings.
      2. Build an HNSW index from *normal* training embeddings.
      3. Query test embeddings → get predictions + confidence.
      4. Vote between branches using confidence (same as original).

    Parameters
    ----------
    x_train, y_train : Tensors
        Training data (already on device).
    x_test : Tensor
        Test data.
    y_test : int or Tensor
        If int → online mode (return predictions + confidence).
        If Tensor → evaluation mode (return metrics).
    model : AE
        The autoencoder.
    k, threshold_percentile, ef_construction, M, ef_search :
        HNSW hyperparameters.

    Returns
    -------
    Online mode  : (predictions, confidence)
    Eval mode    : (result_encoder, result_decoder, result_final, confidence)
    """
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    normal_mask = (y_train_np == 0).squeeze()

    x_train_normal = x_train[normal_mask]

    # ── Encoder branch (layer 0) ──
    enc_normal = _extract_embeddings(model, x_train_normal, 0)
    enc_test = _extract_embeddings(model, x_test, 0)

    det_enc = HNSWAnomalyDetector(
        k=k, threshold_percentile=threshold_percentile,
        ef_construction=ef_construction, M=M, ef_search=ef_search,
        space='cosine')
    det_enc.build(enc_normal)
    pred_enc, conf_enc, dist_enc = det_enc.query(enc_test)

    # ── Decoder branch (layer 1) ──
    dec_normal = _extract_embeddings(model, x_train_normal, 1)
    dec_test = _extract_embeddings(model, x_test, 1)

    det_dec = HNSWAnomalyDetector(
        k=k, threshold_percentile=threshold_percentile,
        ef_construction=ef_construction, M=M, ef_search=ef_search,
        space='cosine')
    det_dec.build(dec_normal)
    pred_dec, conf_dec, dist_dec = det_dec.query(dec_test)

    # ── Vote between branches (same logic as original) ──
    # The branch with higher confidence wins per sample
    y_pred_final = np.where(conf_enc > conf_dec, pred_enc, pred_dec)
    confidence = np.maximum(conf_enc, conf_dec)

    # ── Scoring ──
    if not isinstance(y_test, int):
        y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        result_encoder = score_detail(y_test_np, pred_enc)
        result_decoder = score_detail(y_test_np, pred_dec)
        result_final = score_detail(y_test_np, y_pred_final, if_print=True)
        return result_encoder, result_decoder, result_final, confidence
    else:
        return y_pred_final, confidence
