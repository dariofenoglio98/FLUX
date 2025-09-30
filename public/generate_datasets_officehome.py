
import argparse, os, random, math, itertools, pathlib
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from torchvision.transforms import Compose, Resize, ToTensor
import config as cfg
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--fold",        type=int,   default=123,help="RNG seed")
args = parser.parse_args()

N_CLIENTS     = cfg.n_clients                     # Total FL clients
MAX_LABELS    = cfg.max_labels  # Classes kept per domain (→ relabelled 0…max_labels-1)
TEST_RATIO    = cfg.client_eval_ratio             # Fraction of each client’s data used for testing
SEED          = cfg.random_seed + args.fold                   # RNG seed
OUT_DIR       = pathlib.Path("./data/cur_datasets")  # Where client_*.npy files are written
IMG_DIM       = (64, 64)      # ← NEW: target (H, W) for every image
DOMAINS       = ["Art", "Clipart", "Product", "Real World"]

random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# STEP 1 ── Load & domain-split Office-Home -----------------------------------
# -----------------------------------------------------------------------------
print("🔄  Loading Office-Home…")

if pathlib.Path("officehome_domain_ds").exists():
    domain_ds = DatasetDict.load_from_disk("officehome_domain_ds")
else:
    office_home = load_dataset("flwrlabs/office-home", split="train")     # 15 620 rows

    domain_ds = DatasetDict({
        d: office_home.filter(lambda ex, d=d: ex["domain"] == d)
        for d in DOMAINS
    })
    domain_ds.save_to_disk("officehome_domain_ds")

# -----------------------------------------------------------------------------
# STEP 2 ── Sample label subset & build (old→new) maps ------------------------
# -----------------------------------------------------------------------------
print(f"🎯  Picking {MAX_LABELS} labels per domain & remapping to 0..{MAX_LABELS-1}")
pruned_ds, label_maps = {}, {}          # label_maps[domain] = {old: new}
keep_lbls = list(range(MAX_LABELS))   # ← new name requested

for dom, ds in domain_ds.items():
    # retain examples whose original label is in KEEP_LBLS
    pruned = ds.filter(lambda ex, keep=keep_lbls: ex["label"] in keep)
    pruned_ds[dom] = pruned
    print(f"  • {dom:11s}: {len(pruned):5d} images kept")

# -----------------------------------------------------------------------------
# STEP 3 ── Assign domains → clients (non-IID) --------------------------------
# -----------------------------------------------------------------------------
client_domains = list(itertools.islice(itertools.cycle(DOMAINS), N_CLIENTS))
random.shuffle(client_domains)                     # random one-domain-per-client
dom2cids = defaultdict(list)
for cid, dom in enumerate(client_domains):
    dom2cids[dom].append(cid)

# -----------------------------------------------------------------------------
# STEP 4 ── Helper: convert split to np arrays --------------------------------
# -----------------------------------------------------------------------------
transform = Compose([
    Resize(IMG_DIM),      # ← uses the configurable tuple
    ToTensor()
])

def ds_to_arrays(ds):
    feats, labs = [], []
    for ex in ds:
        feats.append(transform(ex["image"]).numpy())
        labs.append(ex["label"])
    return np.stack(feats), np.asarray(labs, dtype=np.int64)

# -----------------------------------------------------------------------------
# STEP 5 ── Write each client file with train & test tensors ------------------
# -----------------------------------------------------------------------------
print(f"\n📤  Writing client shards  (test_ratio = {TEST_RATIO:.2f})")
for dom, cids in dom2cids.items():
    ds_dom  = pruned_ds[dom]
    n_each  = math.ceil(len(ds_dom) / len(cids))

    idx_all = np.random.permutation(len(ds_dom))
    splits  = [idx_all[i*n_each:(i+1)*n_each] for i in range(len(cids))]

    for cid, idx in zip(cids, splits):
        split_ds = ds_dom.select(idx.tolist())
        feats, labs = ds_to_arrays(split_ds)

        # ------- train / test split
        n_test          = int(len(feats) * TEST_RATIO)
        test_idx        = np.random.permutation(len(feats))[:n_test]
        train_idx       = np.setdiff1d(np.arange(len(feats)), test_idx)

        save_dict = {
            "train_features": torch.from_numpy(feats[train_idx]),
            "train_labels":   torch.from_numpy(labs[train_idx]),
            "test_features":  torch.from_numpy(feats[test_idx]),
            "test_labels":    torch.from_numpy(labs[test_idx]),
        }

        outfile = OUT_DIR / f"client_{cid}.npy"
        np.save(outfile, save_dict, allow_pickle=True)
        print(f"  • client {cid:2d}  ← {dom:11s}  "
              f"(train {len(train_idx):4d} | test {len(test_idx):3d}) → {outfile}")

n_clusters = len(set(client_domains))
print(f"\033[91mNumber of clusters: {n_clusters}\033[0m")
np.save(OUT_DIR / "n_clusters.npy", n_clusters)

print("\n✅  Finished!\n\n")
