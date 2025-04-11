# File: main.py

import os
import csv
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# Local imports
from models import get_model
from train_eval import train_one_epoch, eval_model
from membership_inference import (
    compute_confidence_attack_metrics,
    compute_loss_attack_metrics,
    compute_shadow_attack_metrics
)
from split_ai import SplitAIEnsemble
from self_distillation import get_soft_labels, DistilledDataset, train_distilled_model
from selena_train_eval import train_one_epoch_selena, eval_model_selena
from dyna_noise import DynaNoise

from data_loader import (
    get_sst2_loader, 
    load_imagenet_subset, 
    get_data_loaders
)

###############################################################################
# CSV saving function
###############################################################################
def save_results_to_csv(csv_filename, fieldnames, row_dict):
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_filename) == 0:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"[INFO] Appended results to {csv_filename}")

###############################################################################
# Fix Seeds
###############################################################################
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
# For CIFAR or "ImageNet subset" placeholders, load them with a fixed transform
###############################################################################
def get_fixed_cifar_loader(model_name, batch_size=128, train=True):
    """
    For demonstration, we handle ONLY 'cifar10' here. For 'imagenet-10',
    we will call load_imagenet_subset from data_loader.py instead.
    """
    base_transform = []
    if model_name.lower() in ["alexnet", "resnet18", "vgg16_bn"]:
        base_transform.append(transforms.Resize(224))

    base_transform.append(transforms.ToTensor())
    base_transform.append(
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    )
    transform = transforms.Compose(base_transform)

    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=train,
        download=True,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

###############################################################################
# maybe_train_model
###############################################################################
def maybe_train_model(dataset_name, model_name, epochs, batch_size, device):
    """
    Loads (or trains) the target model for the given dataset:
      - CIFAR-10 => we call get_fixed_cifar_loader (cifar only)
      - ImageNet-10 => we call load_imagenet_subset from data_loader
      - SST-2 => we call get_sst2_loader from data_loader
    """
    ds_lower = dataset_name.lower()
    if ds_lower in ["cifar10", "imagenet-10"]:
        num_classes = 10
    elif ds_lower == "sst2":
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    import os

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(ROOT_DIR, "data")
    save_dir = os.path.join(ROOT_DIR, "saved_models")
    
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_{epochs}epochs.pt")

    model = get_model(model_name, num_classes).to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"[INFO] Loaded model from {ckpt_path}")
        return model

    print(f"[INFO] Training new model: {model_name} on {dataset_name} for {epochs} epochs.")

    # Distinguish dataset sources
    if ds_lower == "cifar10":
        # use the custom function for CIFAR
        train_loader = get_fixed_cifar_loader(model_name, batch_size=batch_size, train=True)
    elif ds_lower == "imagenet-10":
        # use load_imagenet_subset from data_loader
        # returns a single DataLoader with entire dataset => we manually interpret as "train"
        train_loader, _ = load_imagenet_subset(data_dir, dataset_name="imagenet-10", batch_size=batch_size)
    elif ds_lower == "sst2":
        # for sst2
        train_loader = get_sst2_loader(data_dir, batch_size=batch_size, train=True)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for ep in range(epochs):
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, device=device)
        if (ep + 1) % 2 == 0:
            print(f"[EPOCH {ep+1}] train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Model saved to {ckpt_path}")
    return model

###############################################################################
# MIA
###############################################################################
def run_mia_evaluation(model,
                       in_loader, out_loader,
                       sin_loader, sout_loader,
                       device='cuda',
                       threshold_conf=0.9,
                       threshold_loss=0.5,
                       epochs=15,
                       dataset_name='cifar10',
                       model_name='alexnet'):

    conf = compute_confidence_attack_metrics(model, in_loader, out_loader,
                                             threshold=threshold_conf,
                                             device=device)
    loss = compute_loss_attack_metrics(model, in_loader, out_loader,
                                       threshold=threshold_loss,
                                       device=device)
    shadow = compute_shadow_attack_metrics(
        model, in_loader, out_loader,
        sin_loader, sout_loader,
        device=device, epochs=epochs,
        model_name=model_name, dataset_name=dataset_name
    )
    return conf, loss, shadow

###############################################################################
# Evaluate Model with Noise
###############################################################################
def eval_model_with_noise(model, loader, dyna_noise, dataset_name, device='cuda'):
    model.eval()
    ds_lower = dataset_name.lower()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if ds_lower == 'sst2':
                input_ids = batch['input_ids'].to(device).long()
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).long()
                logits = model(input_ids, attention_mask=attention_mask)
                noisy = dyna_noise.forward(logits)
                preds = noisy.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                images, lbl = batch
                images = images.to(device)
                lbl = lbl.to(device).long()
                logs = model(images)
                noisy = dyna_noise.forward(logs)
                preds = noisy.argmax(dim=1)
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)
    return correct / total

###############################################################################
# Main
###############################################################################
def main(dataset_name='imagenet-10', model_name='alexnet', epochs=15, batch_size=64,
         bv=0.5, ls=5.0, t=3.0, seed=42):
    # Fix seeds
    set_all_seeds(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}, seed={seed}")

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(ROOT_DIR, "data")


    # 1) maybe_train_model
    model = maybe_train_model(dataset_name, model_name, epochs, batch_size, device)

    # 2) Evaluate test accuracy
    ds_lower = dataset_name.lower()
    if ds_lower == "cifar10":
        # use get_fixed_cifar_loader in test mode
        test_loader = get_fixed_cifar_loader(model_name, batch_size=batch_size, train=False)
    elif ds_lower == "imagenet-10":
        # call load_imagenet_subset for the test portion
        # note: load_imagenet_subset returns "trainset, testset" splitted internally. We'll just load them
        _, test_loader = load_imagenet_subset(data_dir, dataset_name="imagenet-10", batch_size=batch_size)
    elif ds_lower == "sst2":
        # from data_loader import get_sst2_loader
        test_loader = get_sst2_loader(data_dir, batch_size=batch_size, train=False)
    else:
        # handle or raise an exception for unhandled dataset
        raise ValueError(f"Unknown or unimplemented dataset for test loader: {dataset_name}")

    test_acc_no_def = eval_model(model, test_loader, device=device)
    print(f"[INFO] Test accuracy (no defense) = {test_acc_no_def:.4f}")

    # 3) Build membership splits (with fixed seeds) 

    if ds_lower == "cifar10":
        full_train_loader = get_fixed_cifar_loader(model_name, batch_size=batch_size, train=True)
        ds_full = full_train_loader.dataset
    elif ds_lower == "imagenet-10":
        train_loader_imnet, _ = load_imagenet_subset(data_dir, dataset_name="imagenet-10", batch_size=batch_size)
        ds_full = train_loader_imnet.dataset
    elif ds_lower == "sst2":
        train_loader_sst2 = get_sst2_loader(data_dir='data', batch_size=batch_size, train=True)
        ds_full = train_loader_sst2.dataset
    else:
        raise ValueError(f"Unsupported dataset for building membership splits: {dataset_name}")

    g_main = torch.Generator().manual_seed(seed)
    n_full = len(ds_full)
    target_sz = int(0.7 * n_full)
    shadow_sz = n_full - target_sz

    target_ds, shadow_ds = random_split(ds_full, [target_sz, shadow_sz], generator=g_main)

    g_t = torch.Generator().manual_seed(seed + 1)
    nt = len(target_ds)
    tin_sz = int(0.8 * nt)
    tout_sz = nt - tin_sz
    tin_ds, tout_ds = random_split(target_ds, [tin_sz, tout_sz], generator=g_t)

    g_s = torch.Generator().manual_seed(seed + 2)
    ns = len(shadow_ds)
    sin_sz = ns // 2
    sout_sz = ns - sin_sz
    sin_ds, sout_ds = random_split(shadow_ds, [sin_sz, sout_sz], generator=g_s)

    def build_loader_for_subset(subset, shuffle=True):
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    in_loader   = build_loader_for_subset(tin_ds)
    out_loader  = build_loader_for_subset(tout_ds)
    sin_loader  = build_loader_for_subset(sin_ds)
    sout_loader = build_loader_for_subset(sout_ds)

    # 4) Evaluate MIA
    conf_no_def, loss_no_def, shadow_no_def = run_mia_evaluation(
        model, in_loader, out_loader, sin_loader, sout_loader,
        device=device, epochs=epochs,
        dataset_name=dataset_name, model_name=model_name
    )
    print("[INFO] MIA => No Defense => Confidence:", conf_no_def)
    print("[INFO] MIA => No Defense => Loss:", loss_no_def)
    print("[INFO] MIA => No Defense => Shadow:", shadow_no_def)

    ###########################################################################
    # 5) SELENA 
    ###########################################################################
    K = 25
    L = 10
    sub_ep = 15
    dist_ep = 15

    # number of classes for SELENA
    if ds_lower == "cifar10" or ds_lower == "imagenet-10":
        selena_num_classes = 10
    elif ds_lower == "sst2":
        selena_num_classes = 2
    else:
        raise ValueError("SELENA not configured for this dataset.")

    split_ai = SplitAIEnsemble(
        model_fn=lambda: get_model(model_name, num_classes=selena_num_classes).to(device),
        K=K, L=L, device=device
    )

    # Build in-memory dataset from in_loader
    all_in_samples = []
    for batch in in_loader:
        if ds_lower == 'sst2':
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            bs = labels.size(0)
            for i in range(bs):
                all_in_samples.append({
                    'input_ids': input_ids[i],
                    'attention_mask': attention_mask[i],
                    'label': labels[i]
                })
        else:
            images, labels = batch
            bs = images.size(0)
            for i in range(bs):
                all_in_samples.append((images[i], labels[i]))

    class MemDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, is_sst2=False):
            self.data_list = data_list
            self.is_sst2 = is_sst2
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            if self.is_sst2:
                return {
                    'input_ids': self.data_list[idx]['input_ids'],
                    'attention_mask': self.data_list[idx]['attention_mask'],
                    'label': self.data_list[idx]['label']
                }
            else:
                return self.data_list[idx]

    in_memory_dataset = MemDataset(all_in_samples, is_sst2=(ds_lower=='sst2'))
    split_ai.partition_data(in_memory_dataset)


    base_dir = os.path.abspath(os.path.dirname(__file__))
    folder_path = os.path.join(base_dir, "splitai_submodels")
    all_found = True
    for sub_idx in range(split_ai.K):
        filename = f"splitai_submodel_{sub_idx}_{dataset_name}.pt"
        path = os.path.join(folder_path, filename)
        if not os.path.exists(path):
            all_found = False
            break

    if all_found:
        print("[SplitAI] All sub-model checkpoints found. Loading...")
        split_ai.load_submodels(folder_path, dataset_name=dataset_name)
    else:
        print("[SplitAI] Some sub-model checkpoints not found. Training sub-models now...")
        def sgd_optimizer(params):
            return torch.optim.SGD(params, lr=0.01, momentum=0.9)
        split_ai.train_submodels(
            dataset=in_memory_dataset,
            batch_size=batch_size,
            epochs=sub_ep,
            optimizer_fn=sgd_optimizer,
            train_one_epoch_fn=train_one_epoch_selena
        )
        print("[SplitAI] Training complete. Saving sub-models...")
        split_ai.save_submodels(folder_path, dataset_name=dataset_name)

   
    # Distillation
    
    X_sel, Y_sel = get_soft_labels(split_ai, in_memory_dataset, device=device, batch_size=batch_size)
    dist_data = DistilledDataset(X_sel, Y_sel)

    selena_save_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(selena_save_dir, exist_ok=True)
    selena_ckpt_path = os.path.join(selena_save_dir, f"selena_{model_name}_{dataset_name}_{dist_ep}epochs.pt")

    if os.path.exists(selena_ckpt_path):
        selena_model = get_model(model_name, selena_num_classes).to(device)
        selena_model.load_state_dict(torch.load(selena_ckpt_path))
        print(f"[INFO] Loaded SELENA model from {selena_ckpt_path}")
    else:
        print("[SELENA] SELENA model not found. Training distillation now...")
        selena_model = get_model(model_name, num_classes=selena_num_classes).to(device)
        selena_model = train_distilled_model(
            selena_model, dist_data,
            epochs=dist_ep,
            batch_size=batch_size,
            device=device,
            lr=0.01
        )
        torch.save(selena_model.state_dict(), selena_ckpt_path)
        print(f"[INFO] SELENA model saved to {selena_ckpt_path}")


    selena_test_acc = eval_model_selena(selena_model, test_loader, device=device)

    conf_selena, loss_selena, shadow_selena = run_mia_evaluation(
        selena_model, in_loader, out_loader, sin_loader, sout_loader,
        device=device, epochs=dist_ep, dataset_name=dataset_name, model_name=model_name
    )

    ###########################################################################
    # 6) DynaNoise (post-hoc noise defense)
    ###########################################################################
    dyna = DynaNoise(base_variance=bv, lambda_scale=ls, temperature=t)
    dyna_test_acc = eval_model_with_noise(model, test_loader, dyna, dataset_name=dataset_name, device=device)

    def run_mia_dyna(m, inL, outL, sinL, soutL):
        from membership_inference import (
            compute_confidence_attack_metrics,
            compute_loss_attack_metrics,
            compute_shadow_attack_metrics
        )
        cA = compute_confidence_attack_metrics(m, inL, outL, threshold=0.9, dyna_noise=dyna, device=device)
        lA = compute_loss_attack_metrics(m, inL, outL, threshold=0.5, dyna_noise=dyna, device=device)
        sA = compute_shadow_attack_metrics(
            m, inL, outL, sinL, soutL,
            device=device, dyna_noise=dyna,
            epochs=epochs, model_name=model_name, dataset_name=dataset_name
        )
        return cA, lA, sA

    conf_dyna, loss_dyna, shadow_dyna = run_mia_dyna(model, in_loader, out_loader, sin_loader, sout_loader)

    ############################################################################
    # Summarize + PUT metrics
    ############################################################################
    final_res = {
        "dataset": dataset_name,
        "model": model_name,
        "epochs": epochs,
        "base_variance": bv,
        "lambda_scale": ls,
        "temperature": t,
        "test_acc_no_def": round(test_acc_no_def, 4),
        "conf_acc_no_def": round(conf_no_def["accuracy"], 4),
        "loss_acc_no_def": round(loss_no_def["accuracy"], 4),
        "shadow_acc_no_def": round(shadow_no_def["accuracy"], 4),
        "test_acc_selena": round(selena_test_acc, 4),
        "conf_acc_selena": round(conf_selena["accuracy"], 4),
        "loss_acc_selena": round(loss_selena["accuracy"], 4),
        "shadow_acc_selena": round(shadow_selena["accuracy"], 4),
        "test_acc_dyna": round(dyna_test_acc, 4),
        "conf_acc_dyna": round(conf_dyna["accuracy"], 4),
        "loss_acc_dyna": round(loss_dyna["accuracy"], 4),
        "shadow_acc_dyna": round(shadow_dyna["accuracy"], 4),
    }

    # --- Compute PUT metric for SELENA ---
    acc_drop_selena = final_res["test_acc_no_def"] - final_res["test_acc_selena"]
    conf_imp_selena = final_res["conf_acc_no_def"] - final_res["conf_acc_selena"]
    loss_imp_selena = final_res["loss_acc_no_def"] - final_res["loss_acc_selena"]
    shadow_imp_selena = final_res["shadow_acc_no_def"] - final_res["shadow_acc_selena"]
    avg_imp_selena = (conf_imp_selena + loss_imp_selena + shadow_imp_selena) / 3.0
    put_selena = avg_imp_selena - acc_drop_selena

    # --- Compute PUT metric for DynaNoise ---
    acc_drop_dyna = final_res["test_acc_no_def"] - final_res["test_acc_dyna"]
    conf_imp_dyna = final_res["conf_acc_no_def"] - final_res["conf_acc_dyna"]
    loss_imp_dyna = final_res["loss_acc_no_def"] - final_res["loss_acc_dyna"]
    shadow_imp_dyna = final_res["shadow_acc_no_def"] - final_res["shadow_acc_dyna"]
    avg_imp_dyna = (conf_imp_dyna + loss_imp_dyna + shadow_imp_dyna) / 3.0
    put_dyna = avg_imp_dyna - acc_drop_dyna

    final_res["put_selena"] = round(put_selena, 4)
    final_res["put_dyna"]   = round(put_dyna, 4)

    # Per-Attack PUT for SELENA
    put_selena_conf   = conf_imp_selena   - acc_drop_selena
    put_selena_loss   = loss_imp_selena   - acc_drop_selena
    put_selena_shadow = shadow_imp_selena - acc_drop_selena

    final_res["put_selena_conf"]   = round(put_selena_conf, 4)
    final_res["put_selena_loss"]   = round(put_selena_loss, 4)
    final_res["put_selena_shadow"] = round(put_selena_shadow, 4)

    # Per-Attack PUT for Dyna
    put_dyna_conf   = conf_imp_dyna   - acc_drop_dyna
    put_dyna_loss   = loss_imp_dyna   - acc_drop_dyna
    put_dyna_shadow = shadow_imp_dyna - acc_drop_dyna

    final_res["put_dyna_conf"]   = round(put_dyna_conf, 4)
    final_res["put_dyna_loss"]   = round(put_dyna_loss, 4)
    final_res["put_dyna_shadow"] = round(put_dyna_shadow, 4)

    print("\n=== FINAL RESULTS (INTEGRATED) ===")
    for k, v in final_res.items():
        print(f"{k}: {v}")

    csvf = "results.csv"
    keys = list(final_res.keys())
    save_results_to_csv(csvf, keys, final_res)

if __name__ == "__main__":
    # Example runs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ImageNet-10
    main(dataset_name='imagenet-10',
        model_name='alexnet',
        epochs=15,
        batch_size=64,
        bv=0.1,
        ls=2.0,
        t=6.0,
        seed=42)

