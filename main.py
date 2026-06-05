import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


# These lines allow PyTorch to use TF32 precision for matmuls and convolutions
# on Ampere+ GPUs. Tensors in memory remain FP32 — only the multiply-accumulate
# inside the tensor cores runs at reduced precision.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS = 120
WARMUP_EPOCHS = 15
LR_START = 0.4
# 0.01 makes warmup start at LR=0.004 instead of 0.4, preventing training
# divergence at the large batch-512 learning rate, and giving the network a
# stable 15-epoch window to settle before the pruning prox is activated.
LR_WARMUP_START_FACTOR = 0.01
MOMENTUM = 0.9
NUM_WORKERS = 12

TARGET_SPARSITIES = [0.0, 25.0, 50.0, 75.0, 90.0]
SPARSITY_TOLERANCE = 5.0

# Per-method lambda search range. The element-wise L1 prox shrinks each weight
# by lambda*lr per step, so at LR_START=0.4 even lambda=0.05 wipes whole layers
# in one step — its useful regime is ~1e-6..1e-3. The group-L1 prox shrinks
# per-filter norms (typically ~0.1..1), so its useful regime is much higher.
# The search is done in log-space, so these define a multi-decade band rather
# than a linear interval.
LAMBDA_RANGE_BY_METHOD = {
    "Soft_Thresholding_Conv": (1e-7, 1e-2),
    "Block_Soft_Thresholding_Conv": (1e-3, 1),
}

# Max iterations for the binary search for lambda.
MAX_BISECT_ITER = 20

METHODS = [
    "Soft_Thresholding_Conv",
    "Block_Soft_Thresholding_Conv",
]

SEEDS = [7, 42, 1234]
VAL_SIZE = 10000
SPLIT_SEED = 0

SAVE_DIR = "./training_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ResNet20Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=False,
            trust_repo=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def standard_regularized_conv_names(model: nn.Module) -> list[str]:
    """
    Convs targeted by element-wise soft-thresholding: all non-downsample convs.
    """
    return [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and "downsample" not in name
    ]


def block_regularized_conv_names(model: nn.Module) -> list[str]:
    """
    Each block's first conv (.conv1) only. Excludes .conv2 (its output is
    added to the skip path y = x + F(x), so zeroing a filter doesn't remove
    a channel), downsample convs, and the stem (no .conv2 sibling).
    """
    output: list[str] = []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if (
            not isinstance(module, nn.Conv2d)
            or "downsample" in name
            or not name.endswith(".conv1")
        ):
            continue
        sibling = name[: -len(".conv1")] + ".conv2"
        if sibling not in modules:
            continue
        output.append(name)
    return output


def regularized_conv_names(model: nn.Module, method: str) -> list[str]:
    if method == "Soft_Thresholding_Conv":
        return standard_regularized_conv_names(model)
    if method == "Block_Soft_Thresholding_Conv":
        return block_regularized_conv_names(model)
    raise ValueError(f"Unknown method: {method}")


def _conv_to_bn(
    modules: dict[str, nn.Module], conv_name: str
) -> nn.BatchNorm2d | None:
    """
    BasicBlock convention: <prefix>.conv{1,2} pairs with <prefix>.bn{1,2}.
    """
    for suffix in ("conv1", "conv2"):
        if conv_name.endswith(suffix):
            bn_name = conv_name[: -len(suffix)] + "bn" + suffix[-1]
            bn = modules.get(bn_name)
            return bn if isinstance(bn, nn.BatchNorm2d) else None
    return None


def get_conv_weight_sparsity(model: nn.Module, names: list[str]) -> float:
    """
    Element-wise sparsity: percentage of conv weights that are exactly zero,
    over the regularized set.
    """
    assert names is not None
    modules = dict(model.named_modules())

    total = 0
    zero = 0
    for name in names:
        w = modules[name].weight.data
        # Count every individual weight scalar in this conv's kernel tensor
        # (shape: [out_ch, in_ch, kH, kW] -> out_ch * in_ch * kH * kW scalars).
        total += w.numel()
        # Count how many of those scalars were driven to exactly zero.
        zero += (w == 0.0).sum().item()

    ratio = (zero / total) * 100 if total else 0.0
    return ratio


def get_conv_channel_sparsity(model: nn.Module, names: list[str]) -> float:
    """
    Channel-wise sparsity: percentage of conv output channels that are exactly
    zero (all weights in the filter are zero), over the regularized set.
    """
    assert names is not None
    modules = dict(model.named_modules())

    total_ch = 0
    zero_ch = 0
    for name in names:
        w = modules[name].weight.data
        # Flatten each output filter into a single vector
        # ([out_ch, in_ch*kH*kW]) and take its L2 norm, known as the Frobenius
        # norm. A filter is "dead" iff its whole vector is zero, which iff its
        # norm is zero — one scalar per output channel.
        norms = w.view(w.shape[0], -1).norm(p=2, dim=1)
        # w.shape[0] = number of output channels (filters) in this conv.
        total_ch += w.shape[0]
        # A channel counts as zero only when its norm is exactly 0.
        zero_ch += (norms == 0.0).sum().item()

    ratio = (zero_ch / total_ch) * 100 if total_ch else 0.0
    return ratio


def get_search_metric(method: str, model: nn.Module) -> float:
    names = regularized_conv_names(model, method)
    if method == "Soft_Thresholding_Conv":
        return get_conv_weight_sparsity(model, names)
    if method == "Block_Soft_Thresholding_Conv":
        return get_conv_channel_sparsity(model, names)
    raise ValueError(f"Unknown method: {method}")


def apply_soft_thresholding_conv(
    model: nn.Module, lambd: float, lr: float
) -> None:
    """
    Soft-thresholding: Element-wise L1 prox on the method's regularized conv
    weights.
    """
    # Shrinkage threshold for this step. The L1 proximal operator with
    # strength lambda*lr is: w <- sign(w) * max(|w| - lambda*lr, 0).
    thr = lambd * lr
    modules = dict(model.named_modules())
    names = standard_regularized_conv_names(model)
    with torch.no_grad():
        for name in names:
            w = modules[name].weight.data
            # Pull each weight toward zero by `thr`. Weights with |w| <= thr
            # land exactly on zero (relu clips the negative residual); weights
            # with |w| > thr keep their sign and lose `thr` of magnitude.
            modules[name].weight.data = torch.sign(w) * torch.relu(w.abs() - thr)


def apply_block_soft_thresholding_conv(
    model: nn.Module, lambd: float, lr: float
) -> None:
    """
    Block Soft-thresholding: Group-L1 prox over the regularized convs' output
    filters, plus BN zeroing for fully-killed channels.

    Without the BN zeroing, a killed conv filter still produces a constant
    channel = beta after BN, so the channel is not really pruned. We zero
    gamma and beta (and reset running stats) for any channel whose conv
    filter was driven to exactly zero by this prox step, so the channel's
    output is genuinely zero. As a result the channels can be pruned away
    after training.
    """
    # Shrinkage threshold for this step. The group-L1 proximal operator with
    # strength lambda*lr is, per output filter k:
    #     w_k <- max(1 - lambda*lr / ||w_k||_2, 0) * w_k
    thr = lambd * lr
    modules = dict(model.named_modules())
    names = block_regularized_conv_names(model)
    with torch.no_grad():
        for name in names:
            m = modules[name]
            w = m.weight.data
            # Per-filter L2 norm: flatten each output filter k into a vector
            # of size in_ch*kH*kW and take ||.||_2 -> tensor of shape [out_ch].
            norm = w.view(w.shape[0], -1).norm(p=2, dim=1)
            # Shrinkage factor in [0, 1] per filter. Filters with norm <= thr
            # get scale=0 (fully killed); larger filters get a partial scale
            # that uniformly shrinks every weight in the filter toward zero.
            # The +1e-8 prevents division-by-zero on already-dead filters.
            scale = torch.clamp(1 - thr / (norm + 1e-8), min=0)
            # Broadcast scale across (in_ch, kH, kW) so all weights of filter
            # k are multiplied by the same factor.
            m.weight.data *= scale.view(-1, 1, 1, 1)

            # Filters whose scale collapsed to 0 are now all-zero. The
            # downstream BN would still output `beta` on a zero input, so we
            # also have to neutralize the BN channel.
            killed = scale == 0
            if killed.any():
                bn = _conv_to_bn(modules, name)
                if bn is not None:
                    bn.weight.data[killed] = 0.0
                    bn.bias.data[killed] = 0.0
                    bn.running_mean[killed] = 0.0
                    bn.running_var[killed] = 1.0


def apply_sparsity_method(
    method: str, model: nn.Module, lambd: float, lr: float
) -> None:
    if method == "Soft_Thresholding_Conv":
        apply_soft_thresholding_conv(model, lambd, lr)
    elif method == "Block_Soft_Thresholding_Conv":
        apply_block_soft_thresholding_conv(model, lambd, lr)


def build_datasets() -> tuple[Dataset, Dataset, Dataset]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def _cifar(train: bool, tf: transforms.Compose) -> datasets.CIFAR10:
        return datasets.CIFAR10(
            "./data", train=train, download=True, transform=tf
        )

    train_aug = _cifar(True, tf_train)
    train_clean = _cifar(True, tf_test)
    test_set = _cifar(False, tf_test)

    gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_split, val_split = random_split(
        train_aug, [len(train_aug) - VAL_SIZE, VAL_SIZE], generator=gen
    )

    val_subset = Subset(train_clean, val_split.indices)

    return train_split, val_subset, test_set


def build_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    train_generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=train_generator,
        **common,
    )

    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def top1_correct(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    return (outputs.argmax(dim=1) == labels).float().sum().item()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        out = model(images)

        total_loss += criterion(out, labels).item()
        correct += top1_correct(out, labels)
        total += labels.size(0)

    loss = total_loss / len(loader)
    top_1_acc = 100.0 * correct / total if total else 0.0

    return loss, top_1_acc


def build_training_objects(
    seed: int,
) -> tuple[
    nn.Module,
    optim.Optimizer,
    optim.lr_scheduler.LRScheduler,
    nn.Module,
    torch.amp.GradScaler,
]:
    """
    Construct every stateful object a training run needs, fresh per call:
    model, optimizer, LR scheduler, loss criterion, and AMP grad scaler.
    Called once per (lambda, seed) so each bisection iteration starts from
    an identically-seeded fresh model — making search probes comparable.
    """
    set_seed(seed)
    model = ResNet20Classifier().to(DEVICE)

    # SGD with momentum. No weight decay because the prox is doing the
    # regularisation; adding WD on top would shrink the same weights twice
    # and confuse the sparsity-vs-lambda relationship the bisection relies on.
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR_START,
        momentum=MOMENTUM,
    )

    # Phase 1: linear LR warmup from LR_WARMUP_START_FACTOR*LR_START up to
    # LR_START over WARMUP_EPOCHS epochs. Needed because LR_START=0.4 is too
    # large to apply directly at random init (loss diverges) — and the prox
    # is held off during this same window so pruning doesn't fire on
    # untrained weights.
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=LR_WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )

    # Phase 2: cosine anneal from LR_START down to 0 over the remaining
    # epochs. Ending at LR=0 means the prox shrinkage (lambda*lr) also fades
    # to 0, so the last epochs fine-tune without further killing weights.
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )

    # Stitch the two phases together; the milestone tells SequentialLR to
    # switch from `warmup` to `cosine` exactly at epoch WARMUP_EPOCHS.
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    # Standard classification loss; expects raw logits and applies
    # log-softmax + NLL internally.
    criterion = nn.CrossEntropyLoss()

    # AMP grad scaler: prevents FP16 gradient underflow inside the autocast
    # region by scaling the loss up before .backward() and unscaling before
    # the optimizer step. Required for mixed-precision training to be stable.
    scaler = torch.amp.GradScaler("cuda")

    return model, optimizer, scheduler, criterion, scaler


def train_with_lambda(
    method: str,
    lambd: float,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_generator: torch.Generator,
    tag: str,
) -> dict:
    """
    Run a full EPOCHS training pass at the given lambda. Returns the
    final-epoch metrics, per-epoch history, and achieved sparsity.
    """
    set_seed(seed)
    train_generator.manual_seed(seed)
    model, optimizer, scheduler, criterion, scaler = build_training_objects(
        seed
    )
    reg_names = regularized_conv_names(model, method)

    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0.0
        train_total = 0

        loop = tqdm(
            train_loader, leave=False,
            desc=f"[{tag}] Ep {epoch + 1}/{EPOCHS} (lam={lambd:.5f})",
        )

        for images, labels in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                out = model(images)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch >= WARMUP_EPOCHS and lambd > 0:
                apply_sparsity_method(
                    method, model, lambd,
                    optimizer.param_groups[0]["lr"],
                )

            running_loss += loss.item()
            train_correct += top1_correct(out, labels)
            train_total += labels.size(0)

        val_loss, val_top1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "train_top1": 100.0 * train_correct / train_total,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "weight_sparsity": get_conv_weight_sparsity(model, reg_names),
            "channel_sparsity": get_conv_channel_sparsity(model, reg_names),
        })

    _, test_top1 = evaluate(model, test_loader, criterion)
    achieved = get_search_metric(method, model)

    final = history[-1]
    return {
        "achieved_metric": achieved,
        "test_top1": test_top1,
        "history": history,
        "weight_sp": final["weight_sparsity"],
        "channel_sp": final["channel_sparsity"],
        "state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
    }


def find_lambda_for_target(
    method: str,
    target_sparsity: float,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_generator: torch.Generator,
) -> tuple[float, dict]:
    """
    Binary-search for a lambda that hits target_sparsity.

    The search is done on log10(lambda) — i.e. the midpoint is the
    geometric mean of the current bracket — because sparsity reacts to
    lambda across several orders of magnitude. The per-method bracket
    comes from LAMBDA_RANGE_BY_METHOD.

    Returns the lambda from the iteration whose achieved sparsity was
    closest to target, along with that iteration's full result dict.
    """
    range_lo, range_hi = LAMBDA_RANGE_BY_METHOD[method]
    log_lo, log_hi = math.log10(range_lo), math.log10(range_hi)

    best_lam: float | None = None
    best_err = float("inf")
    best_result: dict | None = None

    print(
        f"    [search] method={method} | target={target_sparsity:.1f}% | "
        f"seed={seed} | lambda in [{range_lo:.1e}, {range_hi:.1e}]",
        flush=True,
    )

    for i in range(MAX_BISECT_ITER):
        log_mid = (log_lo + log_hi) / 2
        mid = 10 ** log_mid
        tag = f"{method}_t{target_sparsity:.0f}_S{seed}_it{i + 1}"
        result = train_with_lambda(
            method, mid, seed,
            train_loader, val_loader, test_loader,
            train_generator, tag,
        )
        achieved = result["achieved_metric"]
        err = achieved - target_sparsity
        print(
            f"      iter {i + 1:2d} | lambda={mid:.3e} | "
            f"achieved={achieved:.2f}% | err={err:+.2f}% | "
            f"test_top1={result['test_top1']:.2f}%",
            flush=True,
        )

        if abs(err) < best_err:
            best_lam = mid
            best_err = abs(err)
            best_result = result

        if abs(err) <= SPARSITY_TOLERANCE:
            return mid, result

        if err < 0:
            log_lo = log_mid
        else:
            log_hi = log_mid

    return best_lam, best_result


def main() -> None:
    t0 = time.time()
    print(
        f"Dataset sizes -> train: {50000 - VAL_SIZE} | "
        f"val: {VAL_SIZE} | test: 10000"
    )
    print("\n" + "=" * 70)
    print("ISO-SPARSITY BENCHMARK  (per-method natural metric)")
    print("  Soft_Thresholding_Conv       : accuracy vs weight sparsity")
    print("  Block_Soft_Thresholding_Conv : accuracy vs channel sparsity")
    print("=" * 70)

    train_set, val_set, test_set = build_datasets()
    train_generator = torch.Generator()
    train_loader, val_loader, test_loader = build_loaders(
        train_set, val_set, test_set, train_generator
    )

    baseline_cache: dict[int, tuple[float, dict]] = {}
    aggregated: list[dict] = []

    for method in METHODS:
        primary_metric = (
            "Weight Sparsity (%)" if method == "Soft_Thresholding_Conv"
            else "Channel Sparsity (%)"
        )

        print(f"\n{'=' * 70}")
        print(f"METHOD: {method}  |  primary metric: {primary_metric}")
        print(f"{'=' * 70}")

        for target in TARGET_SPARSITIES:
            print(f"\n  --- Target {primary_metric}: {target:.1f}% ---")

            top1_list: list[float] = []
            w_sp_list: list[float] = []
            ch_sp_list: list[float] = []
            lambdas_used: list[float] = []

            for seed in SEEDS:
                tag = f"{method}_sp{target:.0f}_S{seed}"

                if target == 0.0:
                    if seed in baseline_cache:
                        best_lam, res = baseline_cache[seed]
                    else:
                        best_lam = 0.0
                        res = train_with_lambda(
                            method, 0.0, seed,
                            train_loader, val_loader, test_loader,
                            train_generator, tag,
                        )
                        baseline_cache[seed] = (best_lam, res)
                else:
                    best_lam, res = find_lambda_for_target(
                        method, target, seed,
                        train_loader, val_loader, test_loader,
                        train_generator,
                    )

                pd.DataFrame(res["history"]).to_csv(
                    f"{SAVE_DIR}/history_{tag}.csv", index=False
                )

                torch.save(
                    {
                        "method": method,
                        "target_sparsity": target,
                        "seed": seed,
                        "lambda": best_lam,
                        "test_top1": res["test_top1"],
                        "weight_sp": res["weight_sp"],
                        "channel_sp": res["channel_sp"],
                        "state_dict": res["state_dict"],
                    },
                    f"{SAVE_DIR}/model_{tag}.pt",
                )

                lambdas_used.append(best_lam)
                top1_list.append(res["test_top1"])
                w_sp_list.append(res["weight_sp"])
                ch_sp_list.append(res["channel_sp"])

                print(
                    f"  [{method}] target={target:.1f}% | seed={seed} | "
                    f"lambda={best_lam:.5f} | "
                    f"w_sp={res['weight_sp']:.2f}% | "
                    f"ch_sp={res['channel_sp']:.2f}% | "
                    f"Top-1={res['test_top1']:.4f}%"
                )

            lam_arr = np.array(lambdas_used)
            aggregated.append({
                "Method": method,
                "Primary Metric": primary_metric,
                f"Target {primary_metric}": target,
                "Lambda (mean+/-std)": (
                    f"{lam_arr.mean():.5f}+/-{lam_arr.std():.5f}"
                ),
                "Test Top-1 (%)": (
                    f"{np.mean(top1_list):.4f}+/-"
                    f"{np.std(top1_list):.4f}"
                ),
                "Weight Sparsity (%)": f"{np.mean(w_sp_list):.2f}",
                "Channel Sparsity (%)": f"{np.mean(ch_sp_list):.2f}",
            })

    n_seeds = len(SEEDS)
    seed_label = "SEED" if n_seeds == 1 else "SEEDS"
    print("\n\n" + "#" * 110)
    print(f"BENCHMARK SUMMARY (AVERAGED OVER {n_seeds} {seed_label})")
    print("#" * 110)
    summary_df = pd.DataFrame(aggregated)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{SAVE_DIR}/iso_sparsity_summary.csv", index=False)

    total_time = (time.time() - t0) / 3600
    print(f"\nTotal time: {total_time:.2f} hours")
    print(f"Results stored in: {SAVE_DIR}")


if __name__ == "__main__":
    main()
