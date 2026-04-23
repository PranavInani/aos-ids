"""
AOC-IDS with HNSW-based Anomaly Detection.

Novel contributions:
1. HNSW vector search replaces Gaussian MLE decision boundary
   — captures complex, non-linear clusters of normal behaviour
   — operates in full embedding space (no lossy 1D projection)
   — O(log N) query time via approximate nearest neighbour search
2. Confidence-aware pseudo-labeling (from novel version)
3. Dynamic temperature scheduling via cosine annealing (from novel version)

The autoencoder architecture, CRC loss, and online training loop remain
unchanged — only the detection/classification boundary is replaced.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from hnsw_utils import evaluate_hnsw, HNSWAnomalyDetector
from utils import AE, CRCLoss, SplitData, load_data, setup_seed, score_detail
import argparse
import warnings
import math

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='AOC-IDS with HNSW vector search anomaly detection')

# ---- Original arguments (same interface) ----
parser.add_argument("--dataset", type=str, default='nsl',
                    choices=['nsl', 'unsw'],
                    help="Dataset to use: 'nsl' (NSL-KDD) or 'unsw' (UNSW-NB15)")
parser.add_argument("--epochs", type=int, default=4,
                    help="Number of initial training epochs")
parser.add_argument("--epoch_1", type=int, default=1,
                    help="Number of retraining epochs per online step")
parser.add_argument("--percent", type=float, default=0.8,
                    help="Fraction of training data held out for online simulation")
parser.add_argument("--sample_interval", type=int, default=2000,
                    help="Chunk size for online streaming")
parser.add_argument("--cuda", type=str, default="0",
                    help="CUDA device ID")

# ---- Confidence & temperature arguments ----
parser.add_argument("--confidence_threshold", type=float, default=0.1,
                    help="Min normalised confidence [0-1] to include pseudo-labeled sample")
parser.add_argument("--initial_temp", type=float, default=0.1,
                    help="Starting temperature for cosine-annealed CRC loss")
parser.add_argument("--min_temp", type=float, default=0.02,
                    help="Final temperature after cosine annealing")
parser.add_argument("--online_temp", type=float, default=0.05,
                    help="Temperature used during online phase")

# ---- HNSW-specific arguments ----
parser.add_argument("--hnsw_k", type=int, default=10,
                    help="Number of nearest neighbours for HNSW anomaly detection")
parser.add_argument("--hnsw_threshold_percentile", type=float, default=95.0,
                    help="Percentile of normal k-NN distances used as decision boundary "
                         "(e.g. 95 → top 5%% flagged)")
parser.add_argument("--hnsw_ef_construction", type=int, default=200,
                    help="HNSW construction-time search width (higher = more accurate index)")
parser.add_argument("--hnsw_M", type=int, default=16,
                    help="HNSW max bi-directional links per node per layer")
parser.add_argument("--hnsw_ef_search", type=int, default=100,
                    help="HNSW query-time search width (higher = more accurate queries)")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
sample_interval = args.sample_interval
cuda_num = args.cuda
confidence_threshold = args.confidence_threshold
initial_temp = args.initial_temp
min_temp = args.min_temp
online_temp = args.online_temp

# HNSW params
hnsw_k = args.hnsw_k
hnsw_threshold_pct = args.hnsw_threshold_percentile
hnsw_ef_construction = args.hnsw_ef_construction
hnsw_M = args.hnsw_M
hnsw_ef_search = args.hnsw_ef_search

bs = 128
seed = 5009
seed_round = 1

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196

# ── Load data ──
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════
for i in range(seed_round):
    setup_seed(seed + i)
    print(f"\n{'='*65}")
    print(f"  HNSW AOC-IDS  |  Seed {seed+i}  |  Dataset: {dataset.upper()}")
    print(f"  HNSW params: k={hnsw_k}  percentile={hnsw_threshold_pct}  "
          f"M={hnsw_M}  ef_c={hnsw_ef_construction}  ef_s={hnsw_ef_search}")
    print(f"{'='*65}")

    online_x_train, online_x_test, online_y_train, online_y_test = \
        train_test_split(x_train, y_train, test_size=percent, random_state=seed + i)

    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    criterion = CRCLoss(device, initial_temp)

    # ==================== Phase 1: Initial training ====================
    model.train()
    for epoch in range(epochs):
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
    criterion.temperature = online_temp
    print(f'\n  [Phase 2] Online training  temp={online_temp}')
    print(f'  [HNSW] Detection via k={hnsw_k} nearest-neighbour search '
          f'(threshold @ {hnsw_threshold_pct}th percentile)')

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train = online_x_train.to(device), online_y_train.to(device)

    x_train_this_epoch = online_x_train.clone()
    x_test_left_epoch = online_x_test.clone().to(device)
    y_train_this_epoch = online_y_train.clone()
    y_train_detection = online_y_train.clone()

    count = 0
    total_added = 0
    total_filtered = 0

    while len(x_test_left_epoch) > 0:
        print('seed = ', (seed + i), ', i = ', count)
        count += 1

        # Chunk the remaining test data
        if len(x_test_left_epoch) < sample_interval:
            x_test_this_epoch = x_test_left_epoch.clone()
            x_test_left_epoch.resize_(0)
        else:
            x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
            x_test_left_epoch = x_test_left_epoch[sample_interval:]

        # [HNSW] Get predictions with confidence via HNSW vector search
        predict_label, confidence = evaluate_hnsw(
            x_train_this_epoch, y_train_detection,
            x_test_this_epoch, 0, model,
            k=hnsw_k,
            threshold_percentile=hnsw_threshold_pct,
            ef_construction=hnsw_ef_construction,
            M=hnsw_M,
            ef_search=hnsw_ef_search)

        y_test_pred_this_epoch = predict_label

        # Always add ALL to evaluation tracking (keeps sizes in sync)
        y_train_detection = torch.cat((
            y_train_detection.to(device),
            torch.tensor(y_test_pred_this_epoch).to(device)))

        # Confidence-aware filtering: replace random flipping
        confident_mask = confidence >= confidence_threshold
        n_confident = int(confident_mask.sum())
        n_total = len(y_test_pred_this_epoch)
        total_added += n_confident
        total_filtered += (n_total - n_confident)

        # For training: confident predictions as-is, uncertain → normal (safe default)
        y_for_training = y_test_pred_this_epoch.copy()
        y_for_training[~confident_mask] = 0

        x_train_this_epoch = torch.cat((
            x_train_this_epoch.to(device), x_test_this_epoch.to(device)))
        y_train_this_epoch = torch.cat((
            y_train_this_epoch.to(device),
            torch.tensor(y_for_training).to(device)))

        if count % 10 == 1:
            print(f'  [Online] step {count}: chunk={n_total}, confident={n_confident} '
                  f'({n_confident/max(n_total,1)*100:.1f}%), train_pool={len(x_train_this_epoch)}')

        # Retrain model
        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
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

    print(f'\n  [Summary] Total confident: {total_added}, '
          f'Uncertain (defaulted to normal): {total_filtered} '
          f'({total_filtered/max(total_added+total_filtered,1)*100:.1f}%)')

    # ==================== Final evaluation ====================
    print(f'\n  [Final Evaluation — HNSW-based]')
    res_en, res_de, res_final, _ = evaluate_hnsw(
        x_train_this_epoch, y_train_detection,
        x_test, y_test, model,
        k=hnsw_k,
        threshold_percentile=hnsw_threshold_pct,
        ef_construction=hnsw_ef_construction,
        M=hnsw_M,
        ef_search=hnsw_ef_search)

    print(f'  Encoder  -> Acc={res_en[0]:.4f} Prec={res_en[1]:.4f} '
          f'Rec={res_en[2]:.4f} F1={res_en[3]:.4f}')
    print(f'  Decoder  -> Acc={res_de[0]:.4f} Prec={res_de[1]:.4f} '
          f'Rec={res_de[2]:.4f} F1={res_de[3]:.4f}')
    print(f'  Combined -> Acc={res_final[0]:.4f} Prec={res_final[1]:.4f} '
          f'Rec={res_final[2]:.4f} F1={res_final[3]:.4f}')
