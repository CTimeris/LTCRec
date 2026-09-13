import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def is_ode_model(args):
    return getattr(args, "model_code", "ltcrec") in ["ltcrec_ode", "ltcrec_adode"]


def is_adaptive_ode_model(args):
    return getattr(args, "model_code", "ltcrec") == "ltcrec_adode"


def build_mlp(input_size, hidden_size, output_size, activation="gelu", layers=2, dropout=0.0):
    activation_map = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    act = activation_map[activation]
    modules = []
    in_size = input_size
    for _ in range(max(1, layers - 1)):
        modules.extend([nn.Linear(in_size, hidden_size), act()])
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        in_size = hidden_size
    modules.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*modules)


class LTCRecCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.backbone = build_mlp(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            output_size=hidden_size * 2,
            activation="gelu",
            layers=2,
        )
        self.ff_H = nn.Linear(hidden_size * 2, hidden_size)
        self.ff_G = nn.Linear(hidden_size * 2, hidden_size)
        self.decay_layer = nn.Linear(hidden_size * 2, hidden_size)

        nn.init.xavier_normal_(self.decay_layer.weight, gain=0.1)
        nn.init.zeros_(self.decay_layer.bias)

    def forward(self, item_state, hidden_state, delta_t):
        features = self.backbone(torch.cat([item_state, hidden_state], dim=-1))
        H_k = torch.tanh(self.ff_H(features))
        G_k = torch.tanh(self.ff_G(features))

        decay_rate = F.softplus(self.decay_layer(features))
        z = torch.exp(-decay_rate * delta_t)

        return G_k * z + H_k * (1.0 - z)


class LTCRecODECell(nn.Module):
    def __init__(self, hidden_size, ode_unfolds=3, epsilon=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.ode_unfolds = max(1, ode_unfolds)
        self.epsilon = epsilon
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.steady_state = nn.Parameter(torch.zeros(hidden_size))
        self.stimulus = build_mlp(
            input_size=hidden_size * 2 + 1,
            hidden_size=hidden_size * 2,
            output_size=hidden_size,
            activation="gelu",
            layers=2,
        )

    def forward(self, item_state, hidden_state, delta_t):
        delta_t = delta_t.clamp_min(0.0)
        step_t = delta_t / self.ode_unfolds
        inv_tau = 1.0 / (F.softplus(self.log_tau).view(1, -1) + self.epsilon)
        steady_state = self.steady_state.view(1, -1)
        state = hidden_state
        for _ in range(self.ode_unfolds):
            stimulus_input = torch.cat([item_state, state, step_t], dim=-1)
            g = F.softplus(self.stimulus(stimulus_input))
            rate = inv_tau + g
            update = step_t * (g * steady_state - rate * state)
            state = state + update / (1.0 + step_t * rate + self.epsilon)
        return state


class LTCRecAdaptiveODECell(nn.Module):
    def __init__(self, hidden_size,
                 atol=1e-3, rtol=1e-4,
                 min_step=1e-4, max_step=1.0,
                 max_steps=50, epsilon=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.atol = atol
        self.rtol = rtol
        self.min_step = min_step
        self.max_step = max_step
        self.max_steps = max_steps
        self.epsilon = epsilon

        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.steady_state = nn.Parameter(torch.zeros(hidden_size))
        self.stimulus = build_mlp(
            input_size=hidden_size * 2 + 1,
            hidden_size=hidden_size * 2,
            output_size=hidden_size,
            activation="gelu",
            layers=2,
        )

    def _single_step(self, item_state, state, dt, inv_tau, steady_state):
        stimulus_input = torch.cat([item_state, state, dt], dim=-1)  # (B, 2D+1)
        g = F.softplus(self.stimulus(stimulus_input))                # (B, D)
        rate = inv_tau + g                                           # (B, D)
        update = dt * (g * steady_state - rate * state)              # (B, D)
        return state + update / (1.0 + dt * rate + self.epsilon)     # (B, D)

    def forward(self, item_state, hidden_state, delta_t):
        B, D = hidden_state.shape
        device = hidden_state.device
        dtype = hidden_state.dtype
        delta_t = delta_t.clamp_min(self.min_step)                   # (B, 1)
        inv_tau = 1.0 / (F.softplus(self.log_tau).view(1, -1) + self.epsilon)   # (1, D)
        steady_state = self.steady_state.view(1, -1)                            # (1, D)
        state = hidden_state                                          # (B, D)
        t_current = torch.zeros_like(delta_t)                         # (B, 1)
        dt = torch.minimum(
            delta_t,
            torch.full_like(delta_t, self.max_step)
        )                                                             # (B, 1)
        active = (t_current < delta_t)                                # (B, 1) bool

        for _ in range(self.max_steps):
            if not active.any():
                break
            remaining = delta_t - t_current                           # (B, 1)
            dt_eff = torch.where(active, torch.minimum(dt, remaining), dt)
            dt_eff = dt_eff.clamp_min(self.min_step)
            state_one = self._single_step(
                item_state, state, dt_eff, inv_tau, steady_state)     # (B, D)
            state_half = self._single_step(
                item_state, state, dt_eff / 2, inv_tau, steady_state)
            state_two = self._single_step(
                item_state, state_half, dt_eff / 2, inv_tau, steady_state)

            error = torch.abs(state_two - state_one)                  # (B, D)
            tol = self.atol + self.rtol * torch.max(
                torch.abs(state_two), torch.abs(state_one))           # (B, D)
            error_ratio = (error / tol).max(dim=-1, keepdim=True).values  # (B, 1)
            accept = (error_ratio <= 1.0) & active                    # (B, 1)
            new_t = t_current + dt_eff
            t_current = torch.where(accept, new_t, t_current)
            state = torch.where(accept, state_two, state)
            grow = torch.minimum(dt_eff * 1.5,
                                 torch.full_like(dt_eff, self.max_step))
            shrink = torch.maximum(dt_eff * 0.5,
                                   torch.full_like(dt_eff, self.min_step))
            new_dt = torch.where(
                accept & (error_ratio < 0.2),
                grow,
                torch.where(accept, dt_eff, shrink)
            )
            dt = torch.where(active, new_dt, dt)
            active = (t_current < delta_t - self.epsilon)             # (B, 1)

        return state


class LTCRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_units
        self.model_code = args.model_code

        if self.model_code == 'ltcrec':
            self.cell = LTCRecCell(hidden_size=self.hidden_size)
        elif self.model_code == 'ltcrec_ode':
            self.cell = LTCRecODECell(
                hidden_size=self.hidden_size,
                ode_unfolds=args.ode_unfolds,
                epsilon=args.ltc_epsilon,
            )
        elif self.model_code == 'ltcrec_adode':
            self.cell = LTCRecAdaptiveODECell(
                hidden_size=self.hidden_size,
                atol=args.adode_atol,
                rtol=args.adode_rtol,
                min_step=args.adode_min_step,
                max_step=args.adode_max_step,
                max_steps=args.adode_max_steps,
                epsilon=args.ltc_epsilon,
            )
        else:
            raise ValueError(f"Unknown model code: {self.model_code}")

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask, timespans=None):
        batch_size, seq_len, _ = x.size()
        hidden_state = x.new_zeros((batch_size, self.hidden_size))
        outputs = []
        if timespans is None:
            timespans = x.new_ones((batch_size, seq_len))
        else:
            timespans = timespans.to(device=x.device, dtype=x.dtype)
        for step in range(seq_len):
            valid = mask[:, step].unsqueeze(-1) if mask is not None else None
            item_state = x[:, step]
            delta_t = timespans[:, step].unsqueeze(-1)
            next_state = self.cell(item_state, hidden_state, delta_t)
            hidden_state = torch.where(valid, next_state, hidden_state) if valid is not None else next_state
            outputs.append(hidden_state)
        output = torch.stack(outputs, dim=1)
        return self.layer_norm(self.dropout(output))


class LTCRecBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sequence_layer = LTCRecLayer(args)

    def forward(self, x, mask, timespans=None):
        return self.sequence_layer(x, mask, timespans=timespans)


class LTCRecEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.blocks = nn.ModuleList([LTCRecBlock(args) for _ in range(args.num_blocks)])
        self.bias = nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None, timespans=None):
        for block in self.blocks:
            x = block(x, mask, timespans=timespans)
        if self.args.dataset_code != "xlong":
            return torch.matmul(x, embedding_weight.t()) + self.bias, None
        if labels is None:
            raise ValueError("labels are required for xlong sampled prediction")
        if self.training:
            num_samples = self.args.negative_sample_size
            samples = torch.randint(1, self.args.num_items + 1, size=(*x.shape[:2], num_samples), device=labels.device)
            all_items = torch.cat([samples, labels.unsqueeze(-1)], dim=-1)
            sampled_embeddings = embedding_weight[all_items]
            scores = torch.einsum("b l d, b l i d -> b l i", x, sampled_embeddings) + self.bias[all_items]
            sampled_labels = torch.full_like(labels, num_samples)
            return scores, sampled_labels
        num_samples = self.args.xlong_negative_sample_size
        samples = torch.randint(1, self.args.num_items + 1, size=(x.shape[0], num_samples), device=labels.device)
        all_items = torch.cat([samples, labels], dim=-1)
        sampled_embeddings = embedding_weight[all_items]
        scores = torch.einsum("b l d, b i d -> b l i", x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
        sampled_labels = torch.full_like(labels, num_samples)
        return scores, sampled_labels.reshape(labels.shape)


class LTCRecEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token = nn.Embedding(args.num_items + 1, args.hidden_units)
        self.layer_norm = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        mask = x > 0
        return self.layer_norm(self.dropout(self.token(x))), mask


class LTCRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = LTCRecEmbedding(args)
        self.encoder = LTCRecEncoder(args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0.0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            low = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
            high = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0
            for name, param in self.named_parameters():
                if "layer_norm" in name:
                    continue
                param.uniform_(2 * low - 1, 2 * high - 1)
                param.erfinv_()
                param.mul_(std * math.sqrt(2.0)).add_(mean)

    def forward(self, x, labels=None, timespans=None):
        x, mask = self.embedding(x)
        return self.encoder(x, self.embedding.token.weight, mask, labels=labels, timespans=timespans)

