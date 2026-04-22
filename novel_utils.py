"""
Novel utilities for AOC-IDS with:
1. GMM-based decision boundary (replaces rigid 2-Gaussian MLE)
2. Confidence score output for pseudo-label filtering
"""
from utils import *
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import numpy as np


def evaluate_gmm(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model,
                 max_components=5, get_confidence=False):
    """
    GMM-based evaluation replacing the rigid 2-Gaussian assumption.
    Uses BIC to auto-select optimal number of components.
    """
    num_of_layer = 0
    num_of_output = 1

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    # --- Encoder branch ---
    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    sim_all_en = F.cosine_similarity(train_features, normal_temp.unsqueeze(0), dim=1).cpu().detach().numpy().reshape(-1, 1)
    sim_test_en = F.cosine_similarity(test_features, normal_temp.unsqueeze(0), dim=1).cpu().detach().numpy().reshape(-1, 1)

    y_train_np = y_train.cpu().detach().numpy() if torch.is_tensor(y_train) else np.array(y_train)

    gmm_en = _fit_best_gmm(sim_all_en, max_components)
    pred_en, conf_en = _classify_with_gmm(gmm_en, sim_all_en, y_train_np, sim_test_en)

    # --- Decoder branch ---
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    sim_all_de = F.cosine_similarity(train_recon, normal_recon_temp.unsqueeze(0), dim=1).cpu().detach().numpy().reshape(-1, 1)
    sim_test_de = F.cosine_similarity(test_recon, normal_recon_temp.unsqueeze(0), dim=1).cpu().detach().numpy().reshape(-1, 1)

    gmm_de = _fit_best_gmm(sim_all_de, max_components)
    pred_de, conf_de = _classify_with_gmm(gmm_de, sim_all_de, y_train_np, sim_test_de)

    # --- Vote: pick whichever branch is more confident ---
    pred_final = np.where(conf_en > conf_de, pred_en, pred_de).astype("int32")
    conf_final = np.maximum(conf_en, conf_de).astype("float32")

    if not isinstance(y_test, int):
        y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) and y_test.device != torch.device("cpu") else (y_test.numpy() if torch.is_tensor(y_test) else y_test)
        result_encoder = score_detail(y_test_np, pred_en)
        result_decoder = score_detail(y_test_np, pred_de)
        result_final = score_detail(y_test_np, pred_final, if_print=True)
        if get_confidence:
            return result_encoder, result_decoder, result_final, conf_final
        return result_encoder, result_decoder, result_final
    else:
        if get_confidence:
            return pred_final, conf_final
        return pred_final


def _fit_best_gmm(data, max_components=5):
    """Fit GMM with BIC-based component selection."""
    n_samples = len(data)
    max_k = min(max_components, max(n_samples // 10, 2))
    max_k = max(max_k, 2)

    best_bic, best_gmm = np.inf, None
    for k in range(2, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  random_state=42, max_iter=200)
            gmm.fit(data)
            bic = gmm.bic(data)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        except Exception:
            continue

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        best_gmm.fit(data)
    return best_gmm


def _classify_with_gmm(gmm, train_data, train_labels, test_data):
    """
    Classify test data using fitted GMM.
    Identifies which components are normal vs abnormal using training labels.
    Returns (predictions, confidence_scores).
    """
    train_assignments = gmm.predict(train_data)
    n_components = gmm.n_components

    # Determine which components are "normal" based on training label majority
    normal_components = []
    for k in range(n_components):
        mask = train_assignments == k
        if mask.sum() > 0 and (train_labels[mask] == 0).mean() > 0.5:
            normal_components.append(k)

    # Fallback: if no component is normal, pick the one with highest mean (most similar to normal template)
    if len(normal_components) == 0:
        normal_components = [np.argmax(gmm.means_.flatten())]

    test_proba = gmm.predict_proba(test_data)
    p_normal = np.sum(test_proba[:, normal_components], axis=1)
    p_abnormal = 1.0 - p_normal

    predictions = (p_abnormal > p_normal).astype("int32")
    confidence = np.abs(p_normal - p_abnormal).astype("float32")

    return predictions, confidence
