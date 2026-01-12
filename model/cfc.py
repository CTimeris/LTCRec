import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate
        return output_state, new_cell


class LTCCell(nn.Module):
    def __init__(
        self,
        in_features,
        units,
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self.softplus = nn.Identity()  # 保持与原实现一致
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )
        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )
        # 补充输出映射参数（原代码可能遗漏）
        self._params["output_w"] = self.add_weight(
            name="output_w",
            init_value=torch.ones((self.state_size,)),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            init_value=torch.zeros((self.state_size,)),
        )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # 处理输入感官神经元
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # ODE求解迭代
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / ((elapsed_time + 1) / self._ode_unfolds)
        for t in range(self._ode_unfolds):
            w_activation = self.softplus(self._params["w"]) * self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            rev_activation = w_activation * self._params["erev"]

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (cm_t * v_pre + self.softplus(self._params["gleak"]) * self._params["vleak"] + w_numerator)
            denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator
            v_pre = numerator / (denominator + self._epsilon)
        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"] + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        return state * self._params["output_w"] + self._params["output_b"]

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts):
        ts = ts.view((-1, 1))
        inputs = self._map_inputs(input)
        next_state = self._ode_solver(inputs, hx, ts)
        return next_state


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = hparams.get("no_gate", False)
        self._minimal = hparams.get("minimal", False)

        activation_map = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "lecun": LeCun
        }
        backbone_activation = activation_map[hparams["backbone_activation"]]

        # 主干网络
        layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            backbone_activation(),
        ]
        for i in range(1, self.hparams["backbone_layers"]):
            layer_list.append(nn.Linear(self.hparams["backbone_units"], self.hparams["backbone_units"]))
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys():
                layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)

        if self._minimal:
            self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_size), requires_grad=True)
            self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_size), requires_grad=True)
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):
        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        x = torch.cat([input, hx], 1)

        x = self.backbone(x)
        if self._minimal:
            ff1 = self.ff1(x)
            new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A)
        else:
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_interp = self.sigmoid(self.time_a(x) * ts + self.time_b(x))
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden


# 支持CFC/LTC切换的层
class CfcLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        self.use_ltc = args.use_ltc  # 切换参数

        if self.use_ltc:
            # 初始化LTC细胞
            self.ltc_cell = LTCCell(
                in_features=self.hidden_size,
                units=self.hidden_size,
                ode_unfolds=args.ode_unfolds,  # 从参数获取
                epsilon=args.ltc_epsilon           # 从参数获取
            )
        else:
            # 初始化CFC细胞
            self.ltc_cell = CfcCell(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                hparams={
                    "backbone_activation": "gelu",
                    "backbone_units": self.hidden_size * 2,
                    "backbone_layers": 2,
                    "no_gate": False,
                    "minimal": False
                }
            )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.bert_dropout)

    def forward(self, x, mask, timespans=None):
        batch_size, seq_len, hidden_dim = x.size()
        h_state = torch.zeros((batch_size, self.hidden_size), device=x.device)
        output_sequence = []
        timespans = timespans.to(x.device) if timespans is not None else torch.ones(batch_size, seq_len, device=x.device)

        for t in range(seq_len):
            # 掩码处理
            if mask is not None:
                mask_t = mask[:, t].unsqueeze(1)
                input_t = x[:, t] * mask_t
            else:
                input_t = x[:, t]

            # 统一调用（CFC/LTC的forward参数兼容）
            h_state = self.ltc_cell(input_t, h_state, timespans[:, t])
            output_sequence.append(h_state)

        output = torch.stack(output_sequence, dim=1)
        output = self.dropout(output) # + x  # 残差连接

        # 添加参数约束（仅对LTC生效）
        if self.use_ltc:
            self.ltc_cell.apply_weight_constraints()

        return self.layer_norm(output)


class CfcBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cfc_layer = CfcLayer(args)
        # self.feed_forward = PositionwiseFeedForward(
        #     d_model=args.bert_hidden_units,
        #     d_ff=args.bert_hidden_units * 4,
        #     dropout=args.bert_dropout
        # )

    def forward(self, x, mask, timespans=None):
        x = self.cfc_layer(x, mask, timespans=timespans)
        # x = self.feed_forward(x)
        return x


class CfcModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        layers = args.bert_num_blocks
        self.cfc_blocks = nn.ModuleList([CfcBlock(self.args) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None, timespans=None):
        for cfc_block in self.cfc_blocks:
            x = cfc_block(x, mask, timespans=timespans)

        if self.args.dataset_code != 'xlong':
            scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
            return scores, None
        else:
            assert labels is not None
            if self.training:
                num_samples = self.args.negative_sample_size
                samples = torch.randint(1, self.args.num_items + 1, size=(*x.shape[:2], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size
                samples = torch.randint(1, self.args.num_items + 1, size=(x.shape[0], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[
                    all_items.unsqueeze(1)]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_.reshape(labels.shape)


class CFC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = CfcEmbedding(self.args)
        self.model = CfcModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, x, labels=None, timespans=None):
        x, mask = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels, timespans=timespans)


class CfcEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units

        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        x_emb = self.token(x)
        return self.layer_norm(self.embed_dropout(x_emb)), mask


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)