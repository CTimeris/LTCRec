import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import args, set_template, STATE_DICT_KEY
from dataloader import dataloader_factory
from model import LTCRec

args.dataset_code = 'ml-1m'
args.model_code = 'ltcrec'
set_template(args)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", args.device)

SAMPLE_EVERY = 5
ODE_STEPS = 3

train_loader, val_loader, test_loader = dataloader_factory(args)
print("num_users:", args.num_users, "num_items:", args.num_items)

model = LTCRec(args)
ckpt_path = "experiments/ltcrec/ml-1m_0.01_0.3_0.3/models/best_acc_model.pth"
state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=False)[STATE_DICT_KEY]
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()
print("Model loaded from", ckpt_path)

layer = model.encoder.blocks[0].sequence_layer
cell = layer.cell
hidden_size = args.hidden_units


def solve_ode_semi_implicit(H_k, G_k, decay_rate, delta_t, steps=3):
    dt = delta_t / steps
    h = G_k
    for _ in range(steps):
        # h_new = (h + dt * rate * H) / (1 + dt * rate)
        h = (h + dt * decay_rate * H_k) / (1.0 + dt * decay_rate)
    return h


all_deltas = []
all_errors = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if SAMPLE_EVERY > 1 and (batch_idx % SAMPLE_EVERY != 0):
            continue

        seqs, labels, timespans = batch
        seqs = seqs.to(args.device)
        timespans = timespans.to(args.device)

        x, mask = model.embedding(seqs)
        B, L, D = x.shape
        hidden_state = x.new_zeros((B, hidden_size))

        for step in range(L):
            valid = mask[:, step].unsqueeze(-1)
            item_state = x[:, step]
            delta_t = timespans[:, step].unsqueeze(-1)

            features = cell.backbone(torch.cat([item_state, hidden_state], dim=-1))
            H_k = torch.tanh(cell.ff_H(features))
            G_k = torch.tanh(cell.ff_G(features))
            decay_rate = F.softplus(cell.decay_layer(features))

            z = torch.exp(-decay_rate * delta_t)
            h_closed = G_k * z + H_k * (1.0 - z)

            h_ode = solve_ode_semi_implicit(H_k, G_k, decay_rate, delta_t, steps=ODE_STEPS)

            denom = torch.clamp(torch.norm(h_closed, dim=-1), min=1e-2)
            error = torch.norm(h_closed - h_ode, dim=-1) / denom

            valid_flat = valid.squeeze(-1)
            if valid_flat.any():
                error_valid = error[valid_flat]
                delta_valid = delta_t.squeeze(-1)[valid_flat]
                all_errors.extend(error_valid.cpu().numpy().tolist())
                all_deltas.extend(delta_valid.cpu().numpy().tolist())

            hidden_state = torch.where(valid, h_closed, hidden_state)

        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx+1} batches, collected {len(all_errors)} samples")

deltas = np.array(all_deltas, dtype=np.float64)
errors = np.array(all_errors, dtype=np.float64)

if len(deltas) == 0:
    print("No valid samples collected.")
    sys.exit(0)

deltas_min = deltas / 60.0

bins = [0, 10, 60, 1440, 10080, np.inf]
labels = ["<10 min", "10 min-1 h", "1 h-1 d", "1 d-7 d", ">7 d"]

df = pd.DataFrame({"delta_min": deltas_min, "error": errors})
df["bin"] = pd.cut(df["delta_min"], bins=bins, labels=labels, right=False)

counts = df.groupby("bin", observed=False).size()
proportion = counts / counts.sum() * 100

finite = np.isfinite(errors)
df_err = df[finite]
mean_err = df_err.groupby("bin", observed=False)["error"].mean() * 100

result = pd.DataFrame({
    "mean_error_%": mean_err,
    "proportion_%": proportion,
    "count": counts,
}).reindex(labels)

print("\n===== Approximation Error by Interval =====")
print(result.round(4))
print("\nOverall mean error: {:.4f}%".format(
    np.nanmean(errors[np.isfinite(errors)]) * 100))
print("Total valid samples:", len(df))
print("Finite error samples:", int(finite.sum()))