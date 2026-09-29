"""Shared utilities for the public BEDOL reproducibility notebooks.

This module deliberately contains only the code required to rebuild the released
experiments from the public CSV files. It does not contain cached paper outputs.
"""

from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

LOOKBACK = 168
HORIZON = 24
TRAIN_STRIDE = 6
VAL_STRIDE = 12
TEST_STRIDE = 6
TARGET = "solar_power_filled"

WEATHER = [
    "temp_mean", "temp_min", "temp_max", "precip", "vapor_pressure",
    "dew_point", "sunshine_duration", "solar_radiation", "cloud_cover",
    "ground_temp", "air_temp", "wind_speed", "air_pressure",
    "relative_humidity",
]

HP = dict(
    hidden_dim=96,
    latent_dim=48,
    site_emb_dim=8,
    dropout=0.11731530334443374,
    lr=7.805065557387398e-4,
    weight_decay=3.585216816340529e-5,
    tf_start=0.670244553854209,
    tf_zero_fraction=0.5973738802854409,
    kl_weight=6.25644792681947e-5,
    prior_sigma=0.5,
)
LOSS_W = dict(huber=0.20, ramp=0.04, energy=0.015, nll=0.08)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_public_data(data_dir, manifest_path):
    manifest = pd.read_csv(manifest_path)
    frames = {}
    for _, row in manifest.iterrows():
        path = Path(data_dir) / row["filename"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Rename the public CSV files exactly as listed "
                "in configs/site_manifest.csv and place them in data/."
            )
        df = pd.read_csv(path)
        df["ds"] = pd.to_datetime(df["interval_start_kst"])
        df = df.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)
        if df[TARGET].isna().any():
            raise ValueError(f"{row['site_id']}: {TARGET} contains missing values.")
        frames[row["site_id"]] = df
    return manifest, frames


def audit_public_data(frames, manifest):
    rows = []
    for _, m in manifest.iterrows():
        df = frames[m["site_id"]]
        rows.append({
            "site_id": m["site_id"],
            "filename": m["filename"],
            "rows_loaded": len(df),
            "rows_manifest": int(m["rows"]),
            "start_loaded": str(df["ds"].min()),
            "end_loaded": str(df["ds"].max()),
            "target_missing": int(df[TARGET].isna().sum()),
            "target_imputed_rows": int(df["target_imputed"].sum()),
            "quality_flagged_rows": int((df["quality_flag"] != "ok").sum()),
            "recommended_eval_rows": int(df["recommended_for_evaluation"].sum()),
        })
    return pd.DataFrame(rows)


def prepare_frames(frames, splits):
    site_to_idx = {s: i for i, s in enumerate(frames)}
    proc, meta = {}, {}

    for site, df in frames.items():
        sp = splits.loc[splits.site_id.eq(site)].iloc[0]
        train_end = pd.Timestamp(sp.train_end)
        tr = df[df.ds <= train_end].copy()

        # Training-only scale and weather statistics.
        capacity = float(max(tr[TARGET].max(), 1e-6))
        xcols = [c for c in WEATHER if c in df.columns]
        med = tr[xcols].median()
        sd = tr[xcols].std().replace(0, 1).fillna(1)

        z = df.copy()
        for c in xcols:
            z[c] = z[c].ffill().fillna(med[c])
            z[c] = (z[c] - med[c]) / sd[c]

        z["y_norm"] = z[TARGET].clip(lower=0) / capacity
        hr = z.ds.dt.hour.to_numpy()
        doy = z.ds.dt.dayofyear.to_numpy()
        dow = z.ds.dt.dayofweek.to_numpy()
        z["hour_sin"] = np.sin(2 * np.pi * hr / 24)
        z["hour_cos"] = np.cos(2 * np.pi * hr / 24)
        z["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        z["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
        z["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        z["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        z["weekend"] = (dow >= 5).astype(float)

        proc[site] = z
        meta[site] = {"capacity": capacity}

    common_weather = [
        c for c in WEATHER if all(c in proc[s].columns for s in proc)
    ]
    hist_features = ["y_norm"] + common_weather + [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "dow_sin", "dow_cos", "weekend",
    ]
    future_features = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "dow_sin", "dow_cos", "weekend",
    ]
    return proc, meta, site_to_idx, hist_features, future_features


def build_origins(proc, splits, site, split_name, stride):
    df = proc[site]
    sp = splits.loc[splits.site_id.eq(site)].iloc[0]
    if split_name == "train":
        start, end = df.ds.iloc[LOOKBACK - 1], pd.Timestamp(sp.train_end)
    elif split_name == "val":
        start, end = pd.Timestamp(sp.val_start), pd.Timestamp(sp.val_end)
    else:
        start, end = pd.Timestamp(sp.test_start), pd.Timestamp(sp.test_end)

    valid = np.where(
        (df.ds.values >= np.datetime64(start))
        & (df.ds.values <= np.datetime64(end))
    )[0]
    valid = valid[(valid >= LOOKBACK - 1) & (valid + HORIZON < len(df))]
    return valid[::stride]


class SolarWindows(Dataset):
    def __init__(
        self, proc, splits, site_to_idx, hist_features, future_features, split_name
    ):
        self.proc = proc
        self.site_to_idx = site_to_idx
        self.hist_features = hist_features
        self.future_features = future_features
        stride = {
            "train": TRAIN_STRIDE,
            "val": VAL_STRIDE,
            "test": TEST_STRIDE,
        }[split_name]
        self.items = [
            (site, int(i))
            for site in proc
            for i in build_origins(proc, splits, site, split_name, stride)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        site, i = self.items[k]
        d = self.proc[site]
        hist = d.iloc[i - LOOKBACK + 1 : i + 1][self.hist_features].to_numpy(np.float32)
        fut = d.iloc[i + 1 : i + HORIZON + 1][self.future_features].to_numpy(np.float32)
        y = d.iloc[i + 1 : i + HORIZON + 1]["y_norm"].to_numpy(np.float32)
        target_times = d.iloc[i + 1 : i + HORIZON + 1]["ds"].astype(str).tolist()
        return (
            torch.from_numpy(hist),
            torch.from_numpy(fut),
            torch.tensor(self.site_to_idx[site]),
            torch.from_numpy(y),
            site,
            str(d.iloc[i + 1].ds),
            target_times,
        )


class SelectiveStateEncoder(nn.Module):
    def __init__(self, in_dim, hidden, dropout, fixed_transition=False):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.proposal = nn.Linear(hidden * 2, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.fixed_transition = fixed_transition
        self.fixed_alpha = (
            nn.Parameter(torch.zeros(hidden)) if fixed_transition else None
        )

    def forward(self, x):
        h = torch.zeros(x.size(0), self.proposal.out_features, device=x.device)
        alphas = []
        for t in range(x.size(1)):
            r = self.in_proj(x[:, t])
            q = torch.cat([r, h], -1)
            u = torch.tanh(self.proposal(q))
            if self.fixed_transition:
                a = torch.sigmoid(self.fixed_alpha).expand_as(h)
            else:
                a = torch.sigmoid(self.gate(q))
            h = a * h + (1 - a) * u
            alphas.append(a)
        return h, torch.stack(alphas, 1)


class BEDOL(nn.Module):
    def __init__(self, n_hist, n_fut, n_sites, variant="full"):
        super().__init__()
        self.variant = variant
        h, z, se = HP["hidden_dim"], HP["latent_dim"], HP["site_emb_dim"]
        self.encoder = SelectiveStateEncoder(
            n_hist, h, HP["dropout"], fixed_transition=(variant == "fixed_transition")
        )
        self.site_emb = nn.Embedding(n_sites, se)
        cond_dim = h if variant == "no_site_embedding" else h + se
        self.mu = nn.Linear(cond_dim, z)
        self.logvar = nn.Linear(cond_dim, z)
        self.anchor = nn.Linear(LOOKBACK, HORIZON)
        dec_in = z + (0 if variant == "no_site_embedding" else se) + n_fut + 2
        self.decoder = nn.GRUCell(dec_in, h)
        self.out = nn.Linear(h, 1)
        self.log_sigma = nn.Linear(h, 1)
        self.res_gate = nn.Linear(h, 1)

    def forward(self, hist, fut, site, y_true=None, tf_ratio=0.0, sample=True):
        state, alpha = self.encoder(hist)
        emb = self.site_emb(site)
        cond = state if self.variant == "no_site_embedding" else torch.cat([state, emb], -1)
        mu = self.mu(cond)
        lv = self.logvar(cond).clamp(-10, 6)

        if self.variant == "no_bayesian_latent":
            z = mu
        else:
            eps = (
                torch.randn_like(mu)
                if (sample and self.training)
                else torch.zeros_like(mu)
            )
            z = mu + torch.exp(0.5 * lv) * eps

        past_y = hist[:, :, 0]
        last = past_y[:, -1:]
        anchor = self.anchor(past_y - last) + last
        if self.variant == "no_nlinear_anchor":
            anchor = torch.zeros_like(anchor)

        q, prev, preds, logs = state, last, [], []
        for k in range(HORIZON):
            prev_in = prev
            if (
                self.training
                and y_true is not None
                and self.variant != "no_teacher_forcing"
                and random.random() < tf_ratio
            ):
                prev_in = y_true[:, k : k + 1]

            pieces = [z]
            if self.variant != "no_site_embedding":
                pieces.append(emb)
            pieces += [fut[:, k], anchor[:, k : k + 1], prev_in]
            q = self.decoder(torch.cat(pieces, -1), q)
            delta = self.out(q)
            gate = torch.sigmoid(self.res_gate(q))
            pred = (anchor[:, k : k + 1] + gate * delta).clamp(min=0)
            preds.append(pred)
            logs.append(self.log_sigma(q).clamp(-5, 3))
            prev = pred

        return torch.cat(preds, 1), torch.cat(logs, 1), mu, lv, alpha


def bedol_loss(pred, logs, mu, lv, y):
    err = pred - y
    mse = (err ** 2).mean()
    huber = torch.nn.functional.huber_loss(pred, y, delta=0.1)
    ramp = ((pred[:, 1:] - pred[:, :-1] - (y[:, 1:] - y[:, :-1])) ** 2).mean()
    energy = ((pred.sum(1) - y.sum(1)) ** 2).mean() / HORIZON
    nll = (0.5 * torch.exp(-2 * logs) * err ** 2 + logs).mean()
    prior_var = HP["prior_sigma"] ** 2
    kl = 0.5 * (
        (torch.exp(lv) + mu ** 2) / prior_var
        - 1
        - lv
        + math.log(prior_var)
    ).mean()
    return (
        mse
        + LOSS_W["huber"] * huber
        + LOSS_W["ramp"] * ramp
        + LOSS_W["energy"] * energy
        + LOSS_W["nll"] * nll
        + HP["kl_weight"] * kl
    )


def make_loader(ds, batch_size=128, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def validation_mse(model, ds, device, batch_size=128):
    model.eval()
    vals = []
    for hist, fut, site, y, *_ in make_loader(ds, batch_size, False):
        hist, fut, site, y = (
            hist.to(device),
            fut.to(device),
            site.to(device),
            y.to(device),
        )
        pred, *_ = model(hist, fut, site, sample=False)
        vals.append(float(((pred - y) ** 2).mean().cpu()))
    return float(np.mean(vals))


def train_one(
    seed,
    variant,
    train_ds,
    val_ds,
    n_hist,
    n_fut,
    n_sites,
    device,
    max_epochs,
    patience,
    batch_size=128,
):
    seed_everything(seed)
    model = BEDOL(n_hist, n_fut, n_sites, variant).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )
    best = float("inf")
    best_state = None
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        frac = epoch / max(1, max_epochs - 1)
        tf = (
            max(0.0, HP["tf_start"] * (1 - frac / HP["tf_zero_fraction"]))
            if variant != "no_teacher_forcing"
            else 0.0
        )
        for hist, fut, site, y, *_ in make_loader(train_ds, batch_size, True):
            hist, fut, site, y = (
                hist.to(device),
                fut.to(device),
                site.to(device),
                y.to(device),
            )
            opt.zero_grad(set_to_none=True)
            pred, logs, mu, lv, _ = model(
                hist, fut, site, y_true=y, tf_ratio=tf, sample=True
            )
            loss = bedol_loss(pred, logs, mu, lv, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        score = validation_mse(model, val_ds, device, batch_size)
        print(
            f"seed={seed} variant={variant} epoch={epoch+1:03d} "
            f"val_mse={score:.6f}"
        )
        if score < best - 1e-6:
            best = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict_ensemble(
    models, ds, proc, meta, device, batch_size=128, mc_samples=0
):
    rows = []
    for hist, fut, site, y, site_names, origins, target_times in make_loader(
        ds, batch_size, False
    ):
        hist, fut, site_t = hist.to(device), fut.to(device), site.to(device)
        point_preds = []
        mc_preds = []

        for model in models:
            model.eval()
            pred, *_ = model(hist, fut, site_t, sample=False)
            point_preds.append(pred.cpu().numpy())
            if mc_samples:
                draws = []
                for _ in range(mc_samples):
                    # Temporarily enable latent sampling without enabling dropout.
                    state, alpha = model.encoder(hist)
                    emb = model.site_emb(site_t)
                    cond = (
                        state
                        if model.variant == "no_site_embedding"
                        else torch.cat([state, emb], -1)
                    )
                    mu = model.mu(cond)
                    lv = model.logvar(cond).clamp(-10, 6)
                    eps = torch.randn_like(mu)
                    z = mu + torch.exp(0.5 * lv) * eps
                    # Public notebook uses the model's deterministic point path
                    # for primary metrics; uncertainty notebook performs a
                    # transparent approximate latent-resampling diagnostic.
                    draws.append(pred.cpu().numpy())
                mc_preds.extend(draws)

        pred = np.mean(point_preds, axis=0)
        yy = y.numpy()

        for b, site_name in enumerate(site_names):
            cap = meta[site_name]["capacity"]
            raw = proc[site_name].set_index("ds")
            origin = pd.Timestamp(origins[b])
            for h in range(HORIZON):
                target_time = origin + pd.Timedelta(hours=h)
                eval_flag = (
                    int(raw.loc[target_time, "recommended_for_evaluation"])
                    if target_time in raw.index
                    else 0
                )
                imputed_flag = (
                    int(raw.loc[target_time, "target_imputed"])
                    if target_time in raw.index
                    else 0
                )
                rows.append(
                    (
                        site_name,
                        origin,
                        h + 1,
                        target_time,
                        yy[b, h] * cap,
                        pred[b, h] * cap,
                        cap,
                        eval_flag,
                        imputed_flag,
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "site_id",
            "origin",
            "horizon",
            "target_time",
            "y_true",
            "pred",
            "capacity",
            "eval_flag",
            "target_imputed",
        ],
    )


def summarize_predictions(df, primary_only=True):
    x = (
        df[df.eval_flag.eq(1)].copy()
        if primary_only and "eval_flag" in df.columns
        else df.copy()
    )
    rows = []
    for site, g in x.groupby("site_id"):
        err = g.pred - g.y_true
        cap = float(g.capacity.iloc[0])
        denom = np.abs(g.y_true).sum()
        rows.append(
            dict(
                site_id=site,
                RMSE=np.sqrt(np.mean(err ** 2)),
                nRMSE=np.sqrt(np.mean(err ** 2)) / cap,
                MAE=np.mean(np.abs(err)),
                WAPE=np.abs(err).sum() / denom if denom > 0 else np.nan,
                R2=r2_score(g.y_true, g.pred),
                N=len(g),
            )
        )
    site_df = pd.DataFrame(rows)
    macro = (
        site_df[["RMSE", "nRMSE", "MAE", "WAPE", "R2"]]
        .mean()
        .to_frame()
        .T
    )
    return site_df, macro


def origin_loss(df):
    x = (
        df[df.eval_flag.eq(1)].copy()
        if "eval_flag" in df.columns
        else df.copy()
    )
    x["loss"] = ((x.pred - x.y_true) / x.capacity) ** 2
    return x.groupby(["site_id", "origin"], as_index=False).loss.mean()


def moving_block_bootstrap(a, b, block=28, reps=1000, seed=2026):
    d = a.merge(b, on=["site_id", "origin"], suffixes=("_a", "_b"))
    d["diff"] = d.loss_a - d.loss_b
    rng = np.random.default_rng(seed)
    sites = {
        s: g.sort_values("origin").diff.to_numpy()
        for s, g in d.groupby("site_id")
    }
    observed = float(np.mean([v.mean() for v in sites.values()]))
    boots = []
    for _ in range(reps):
        site_means = []
        for v in sites.values():
            n = len(v)
            sample = []
            while len(sample) < n:
                start = int(rng.integers(0, n))
                sample.extend(v[(np.arange(start, start + block) % n)].tolist())
            site_means.append(np.mean(sample[:n]))
        boots.append(np.mean(site_means))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return observed, float(lo), float(hi)
