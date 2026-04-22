"""
Novel AOC-IDS Online Training with three improvements:
1. Confidence-aware pseudo-labeling (replaces random label flipping)
2. GMM-based decision boundary (replaces rigid 2-Gaussian MLE)
3. Dynamic temperature scheduling (cosine annealing for CRC loss)
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from novel_utils import *
import argparse
import warnings
import math

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='Novel AOC-IDS: confidence-aware pseudo-labels + GMM + dynamic temperature')
# ---- Original arguments (same interface) ----
parser.add_argument("--dataset", type=str, default='nsl')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--flip_percent", type=float, default=0.2,
                    help="(Legacy) Not used in novel version; kept for CLI compatibility")
parser.add_argument("--sample_interval", type=int, default=2000)
parser.add_argument("--cuda", type=str, default="0")

# ---- Novel arguments ----
parser.add_argument("--confidence_threshold", type=float, default=0.3,
                    help="Min confidence to include a pseudo-labeled sample in training")
parser.add_argument("--max_gmm_components", type=int, default=5,
                    help="Max GMM components for BIC-based selection")
parser.add_argument("--initial_temp", type=float, default=0.1,
                    help="Starting temperature for cosine-annealed CRC loss")
parser.add_argument("--min_temp", type=float, default=0.02,
                    help="Final temperature after cosine annealing")
parser.add_argument("--online_temp", type=float, default=0.05,
                    help="Warmer temperature used during online phase to resist noisy pseudo-labels")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
sample_interval = args.sample_interval
cuda_num = args.cuda
confidence_threshold = args.confidence_threshold
max_gmm_components = args.max_gmm_components
initial_temp = args.initial_temp
min_temp = args.min_temp
online_temp = args.online_temp

bs = 128
seed = 5009
seed_round = 5

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196

# ---- Load data (identical to original) ----
if dataset == 'nsl':
    KDDTrain = load_data("NSL_pre_data/PKDDTrain+.csv")
    KDDTest = load_data("NSL_pre_data/PKDDTest+.csv")
    splitter = SplitData(dataset='nsl')
    x_train, y_train = splitter.transform(KDDTrain, labels='labels2')
    x_test, y_test = splitter.transform(KDDTest, labels='labels2')
else:
    UNSWTrain = load_data("UNSW_pre_data/UNSWTrain.csv")
    UNSWTest = load_data("UNSW_pre_data/UNSWTest.csv")
    splitter = SplitData(dataset='unsw')
    x_train, y_train = splitter.transform(UNSWTrain, labels='label')
    x_test, y_test = splitter.transform(UNSWTest, labels='label')

x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

device = torch.device("cuda:" + cuda_num if torch.cuda.is_available() else "cpu")


def get_cosine_temp(epoch, total_epochs, t_max, t_min):
    """Cosine annealing: t_max -> t_min over total_epochs."""
    return t_min + 0.5 * (t_max - t_min) * (1 + math.cos(math.pi * epoch / total_epochs))


# ---- Main training loop (per seed) ----
for i in range(seed_round):
    setup_seed(seed + i)
    print(f"\n{'='*60}")
    print(f"  Seed round {i+1}/{seed_round}  (seed={seed+i})")
    print(f"{'='*60}")

    online_x_train, online_x_test, online_y_train, online_y_test = \
        train_test_split(x_train, y_train, test_size=percent, random_state=seed + i)

    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # [NOVEL] Initialize CRC loss — temperature will be annealed
    criterion = CRCLoss(device, initial_temp)

    # ==================== Phase 1: Initial training ====================
    model.train()
    for epoch in range(epochs):
        # [NOVEL] Cosine anneal temperature: initial_temp -> min_temp
        current_temp = get_cosine_temp(epoch, epochs, initial_temp, min_temp)
        criterion.temperature = current_temp

        if epoch % 50 == 0:
            print(f'  [Phase 1] epoch {epoch}/{epochs}  temp={current_temp:.4f}')

        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            features, recon_vec = model(inputs)
            loss = criterion(features, labels) + criterion(recon_vec, labels)
            loss.backward()
            optimizer.step()

    # ==================== Phase 2: Online training ====================
    # [NOVEL] Switch to warmer temperature for online phase
    criterion.temperature = online_temp
    print(f'\n  [Phase 2] Online training  temp={online_temp}')

    x_train_dev = x_train.to(device)
    x_test_dev = x_test.to(device)
    online_x_train, online_y_train = online_x_train.to(device), online_y_train.to(device)

    # Evaluation pool (all data — used for GMM fitting in evaluate)
    x_eval_pool = online_x_train.clone()
    y_eval_labels = online_y_train.clone()

    # Training pool (only confident data — used for model retraining)
    x_train_pool = online_x_train.clone()
    y_train_pool = online_y_train.clone()

    x_test_left = online_x_test.clone().to(device)

    count = 0
    total_added = 0
    total_filtered = 0

    while len(x_test_left) > 0:
        count += 1

        # Chunk the remaining test data
        if len(x_test_left) < sample_interval:
            x_chunk = x_test_left.clone()
            x_test_left = x_test_left[:0]  # empty
        else:
            x_chunk = x_test_left[:sample_interval].clone()
            x_test_left = x_test_left[sample_interval:]

        # Compute normal templates from initial labeled normal data
        normal_mask = (online_y_train == 0).squeeze()
        normal_temp = torch.mean(
            F.normalize(model(online_x_train[normal_mask])[0], p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(
            F.normalize(model(online_x_train[normal_mask])[1], p=2, dim=1), dim=0)

        # [NOVEL] GMM-based evaluation with confidence scores
        predict_label, confidence = evaluate_gmm(
            normal_temp, normal_recon_temp,
            x_eval_pool, y_eval_labels,
            x_chunk, 0, model,
            max_components=max_gmm_components,
            get_confidence=True)

        # Add ALL predictions to evaluation pool (keeps GMM fitting accurate)
        x_eval_pool = torch.cat((x_eval_pool, x_chunk.to(device)))
        y_eval_labels = torch.cat((y_eval_labels, torch.tensor(predict_label).to(device)))

        # [NOVEL] Confidence-aware filtering: only train on confident predictions
        confident_mask = confidence >= confidence_threshold
        n_confident = confident_mask.sum()
        n_total = len(predict_label)
        total_added += n_confident
        total_filtered += (n_total - n_confident)

        if n_confident > 0:
            x_confident = x_chunk[confident_mask].to(device)
            y_confident = torch.tensor(predict_label[confident_mask]).to(device)

            x_train_pool = torch.cat((x_train_pool, x_confident))
            y_train_pool = torch.cat((y_train_pool, y_confident))

        print(f'  [Online] step {count}: chunk={n_total}, confident={n_confident} '
              f'({n_confident/n_total*100:.1f}%), train_pool={len(x_train_pool)}')

        # Retrain model on confident data only
        if n_confident > 0:
            train_ds = TensorDataset(x_train_pool, y_train_pool)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds, batch_size=bs, shuffle=True)
            model.train()
            for epoch in range(epoch_1):
                for j, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    features, recon_vec = model(inputs)
                    loss = criterion(features, labels) + criterion(recon_vec, labels)
                    loss.backward()
                    optimizer.step()

    print(f'\n  [Summary] Total added: {total_added}, Filtered out: {total_filtered} '
          f'({total_filtered/(total_added+total_filtered)*100:.1f}% rejected)')

    # ==================== Final evaluation ====================
    normal_mask = (online_y_train == 0).squeeze()
    normal_temp = torch.mean(
        F.normalize(model(online_x_train[normal_mask])[0], p=2, dim=1), dim=0)
    normal_recon_temp = torch.mean(
        F.normalize(model(online_x_train[normal_mask])[1], p=2, dim=1), dim=0)

    print(f'\n  [Final Evaluation]')
    res_en, res_de, res_final = evaluate_gmm(
        normal_temp, normal_recon_temp,
        x_eval_pool, y_eval_labels,
        x_test_dev, y_test, model,
        max_components=max_gmm_components)

    print(f'  Encoder  -> Acc={res_en[0]:.4f} Prec={res_en[1]:.4f} Rec={res_en[2]:.4f} F1={res_en[3]:.4f}')
    print(f'  Decoder  -> Acc={res_de[0]:.4f} Prec={res_de[1]:.4f} Rec={res_de[2]:.4f} F1={res_de[3]:.4f}')
    print(f'  Combined -> Acc={res_final[0]:.4f} Prec={res_final[1]:.4f} Rec={res_final[2]:.4f} F1={res_final[3]:.4f}')
