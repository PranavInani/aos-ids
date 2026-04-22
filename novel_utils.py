"""
Novel utilities for AOC-IDS with:
1. Confidence-aware evaluate (extends original 2-Gaussian with confidence scores)
2. Original evaluate preserved for proven accuracy
"""
from utils import *
import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import scipy.optimize as opt


def evaluate_with_confidence(normal_temp, normal_recon_temp, x_train, y_train,
                             x_test, y_test, model):
    """
    Same proven 2-Gaussian decision boundary as the original evaluate(),
    but ALSO returns per-sample confidence scores for pseudo-label filtering.

    Confidence = max(|pdf_normal - pdf_abnormal|) across encoder and decoder branches.

    Returns:
        If y_test is int (online mode): (predictions, confidence_scores)
        Otherwise (eval mode):          (result_encoder, result_decoder, result_final, confidence_scores)
    """
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    # ── Encoder branch ──
    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, _ = torch.sort(F.cosine_similarity(
        train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_normal, _ = torch.sort(F.cosine_similarity(
        train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_abnormal, _ = torch.sort(F.cosine_similarity(
        train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))

    values_features_all_np = values_features_all.cpu().detach().numpy()
    values_features_test = F.cosine_similarity(
        test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

    # Fit encoder Gaussians
    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy())
    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy())

    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial])
    result = opt.minimize(log_likelihood, initial_params,
                          args=(values_features_all_np,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = [float(v) for v in result.x]
    sigma1_fit, sigma2_fit = abs(sigma1_fit), abs(sigma2_fit)

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()
    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2 - pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    # ── Decoder branch ──
    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, _ = torch.sort(F.cosine_similarity(
        train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_normal, _ = torch.sort(F.cosine_similarity(
        train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_abnormal, _ = torch.sort(F.cosine_similarity(
        train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

    values_recon_all_np = values_recon_all.cpu().detach().numpy()
    values_recon_test = F.cosine_similarity(
        test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    # Fit decoder Gaussians
    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy())
    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params,
                          args=(values_recon_all_np,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = [float(v) for v in result.x]
    sigma3_fit, sigma4_fit = abs(sigma3_fit), abs(sigma4_fit)

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()
    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4 - pdf3)).cpu().detach().numpy().astype("float32")

    if not isinstance(y_test, int):
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)

    # ── Vote + confidence ──
    y_test_pred_no_vote = torch.where(
        torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de),
        torch.from_numpy(y_test_pred_2),
        torch.from_numpy(y_test_pred_4))

    # Confidence: the winning branch's |pdf_abnormal - pdf_normal|, normalized
    confidence_raw = np.maximum(y_test_pro_en, y_test_pro_de)
    # Normalize to [0, 1] using percentile-based scaling for stability
    p95 = np.percentile(confidence_raw, 95) if len(confidence_raw) > 0 else 1.0
    if p95 > 0:
        confidence = np.clip(confidence_raw / p95, 0.0, 1.0).astype("float32")
    else:
        confidence = np.ones_like(confidence_raw, dtype="float32")

    if not isinstance(y_test, int):
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
        return result_encoder, result_decoder, result_final, confidence
    else:
        return y_test_pred_no_vote, confidence
