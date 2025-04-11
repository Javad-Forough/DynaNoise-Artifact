import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from models import get_model

###############################################################################
# Calculate metrics
###############################################################################
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'attack_success_rate': acc
    }

###############################################################################
# 1) CONFIDENCE THRESHOLD ATTACK
###############################################################################
def attack_confidence_threshold(probs, threshold=0.9):
    max_probs, _ = torch.max(probs, dim=1)
    preds = (max_probs > threshold).long()
    return preds

def compute_confidence_attack_metrics(
    model,
    in_loader,
    out_loader,
    threshold=0.9,
    dyna_noise=None,
    device='cuda'
):
    model.eval()
    all_preds = []
    all_labels = []

    def forward_pass(batch):
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            images, _ = batch
            images = images.to(device)
            logits = model(images)
        return logits

    with torch.no_grad():
        # in => label=1
        for batch in in_loader:
            if isinstance(batch, dict):
                batch_size = batch["input_ids"].size(0)
            else:
                _, labs = batch
                batch_size = labs.size(0)

            logits = forward_pass(batch)
            if dyna_noise is not None:
                probs = dyna_noise.forward(logits)
            else:
                probs = F.softmax(logits, dim=1)

            preds_batch = attack_confidence_threshold(probs, threshold)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels.extend([1]*batch_size)

        # out => label=0
        for batch in out_loader:
            if isinstance(batch, dict):
                batch_size = batch["input_ids"].size(0)
            else:
                _, labs = batch
                batch_size = labs.size(0)

            logits = forward_pass(batch)
            if dyna_noise is not None:
                probs = dyna_noise.forward(logits)
            else:
                probs = F.softmax(logits, dim=1)

            preds_batch = attack_confidence_threshold(probs, threshold)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels.extend([0]*batch_size)

    return compute_metrics(np.array(all_preds), np.array(all_labels))

###############################################################################
# 2) LOSS THRESHOLD ATTACK
###############################################################################
def attack_loss_threshold(probs, true_labels, threshold=0.5):
    eps = 1e-12
    p_true = torch.gather(probs, 1, true_labels.unsqueeze(1)).squeeze(1)
    p_true = torch.clamp(p_true, min=eps, max=1.0)
    xent = -torch.log(p_true)
    preds = (xent < threshold).long()
    return preds

def compute_loss_attack_metrics(
    model,
    in_loader,
    out_loader,
    threshold=0.5,
    dyna_noise=None,
    device='cuda'
):
    model.eval()
    all_preds = []
    all_labels = []

    def forward_and_labels(batch):
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_ = batch["label"].to(device)
            logits_ = model(input_ids, attention_mask=attention_mask)
        else:
            images, labels_ = batch
            images = images.to(device)
            labels_ = labels_.to(device)
            logits_ = model(images)
        return logits_, labels_

    with torch.no_grad():
        # in => label=1
        for batch in in_loader:
            logits, labels = forward_and_labels(batch)
            if dyna_noise is not None:
                probs = dyna_noise.forward(logits)
            else:
                probs = F.softmax(logits, dim=1)

            preds_batch = attack_loss_threshold(probs, labels, threshold)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels.extend([1]*labels.size(0))

        # out => label=0
        for batch in out_loader:
            logits, labels = forward_and_labels(batch)
            if dyna_noise is not None:
                probs = dyna_noise.forward(logits)
            else:
                probs = F.softmax(logits, dim=1)

            preds_batch = attack_loss_threshold(probs, labels, threshold)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels.extend([0]*labels.size(0))

    return compute_metrics(np.array(all_preds), np.array(all_labels))

###############################################################################
# Feature extraction for Shadow Attack
###############################################################################
def extract_features(model, inputs, labels, device='cuda', dyna_noise=None):
    with torch.no_grad():
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs)
        else:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

        if dyna_noise is not None:
            probs = dyna_noise.forward(logits)
        else:
            probs = F.softmax(logits, dim=1)

    max_conf, _ = torch.max(probs, dim=1)
    eps = 1e-12
    p_true = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
    p_true = torch.clamp(p_true, min=eps)
    xent = -torch.log(p_true)

    feats = [[c.item(), ce.item()] for c, ce in zip(max_conf, xent)]
    return feats

###############################################################################
# 3) SHADOW MODEL ATTACK
###############################################################################
def train_shadow_model(
    loader_in,
    loader_out,
    num_classes,
    epochs=5,
    device='cuda',
    model_name='alexnet',
    dataset_name='cifar10'
):
    """
    Trains a shadow model (same architecture + same #classes).
    """
    import torch.optim as optim
    from train_eval import train_one_epoch

    shadow_folder = "saved_models_shadow"
    if not os.path.exists(shadow_folder):
        os.makedirs(shadow_folder)

    filename = f"{model_name}_{dataset_name}_shadow_{epochs}epochs.pt"
    path = os.path.join(shadow_folder, filename)

    shadow_model = get_model(model_name, num_classes).to(device)

    # If a pre-trained shadow model exists, load it
    if os.path.exists(path):
        shadow_model.load_state_dict(torch.load(path))
        print(f"[INFO] Loaded shadow model from {path}")
        return shadow_model

    optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)
    shadow_model.train()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(shadow_model, loader_in, optimizer, device=device)
        print(f"[SHADOW TRAIN EPOCH {epoch+1}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    torch.save(shadow_model.state_dict(), path)
    print(f"[INFO] Shadow model saved to {path}")
    return shadow_model

def gather_shadow_features(shadow_model, shadow_in_loader, shadow_out_loader,
                           device='cuda', dyna_noise=None):
    X, Y = [], []
    # in => label=1
    for batch in shadow_in_loader:
        if isinstance(batch, dict):
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            labels = batch["label"]
        else:
            inputs, labels = batch

        feats = extract_features(shadow_model, inputs, labels, device=device, dyna_noise=dyna_noise)
        X.extend(feats)
        Y.extend([1]*len(feats))

    # out => label=0
    for batch in shadow_out_loader:
        if isinstance(batch, dict):
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            labels = batch["label"]
        else:
            inputs, labels = batch

        feats = extract_features(shadow_model, inputs, labels, device=device, dyna_noise=dyna_noise)
        X.extend(feats)
        Y.extend([0]*len(feats))

    return np.array(X), np.array(Y)

def train_attack_model(X, Y):
    from sklearn.linear_model import LogisticRegression
    attack_clf = LogisticRegression()
    attack_clf.fit(X, Y)
    return attack_clf

def compute_shadow_attack_metrics(
    model,
    target_in_loader,
    target_out_loader,
    shadow_in_loader,
    shadow_out_loader,
    device='cuda',
    dyna_noise=None,
    epochs=5,
    model_name='alexnet',
    dataset_name='cifar10'
):
    """
    1) Build or load shadow model with the same #classes as main model
    2) Gather shadow features => train logistic regression
    3) Evaluate on target in/out
    """

    ds_lower = dataset_name.lower()

    # Make sure to set num_classes in a way consistent with your main model
    if ds_lower in ['cifar10', 'imagenet-10']:
        num_classes = 10
    elif ds_lower == 'sst2':
        num_classes = 2
    else:
        # fallback
        num_classes = 2

    # 1) Train shadow model
    shadow_model = train_shadow_model(
        shadow_in_loader,
        shadow_out_loader,
        num_classes=num_classes,
        epochs=epochs,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name
    )

    # 2) Gather shadow features
    X_shadow, Y_shadow = gather_shadow_features(shadow_model,
                                                shadow_in_loader,
                                                shadow_out_loader,
                                                device=device,
                                                dyna_noise=dyna_noise)

    #    Train logistic-regression on these shadow features
    attack_clf = train_attack_model(X_shadow, Y_shadow)

    # 3) Evaluate on target
    X_target = []
    Y_target = []

    def local_extract(model_, batch, is_in):
        if isinstance(batch, dict):
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            labels = batch["label"]
        else:
            inputs, labels = batch
        feats = extract_features(model_, inputs, labels, device=device, dyna_noise=dyna_noise)
        return feats, len(feats), labels

    # in => label=1
    for batch in target_in_loader:
        feats, bs, lbls = local_extract(model, batch, is_in=True)
        X_target.extend(feats)
        Y_target.extend([1]*bs)

    # out => label=0
    for batch in target_out_loader:
        feats, bs, lbls = local_extract(model, batch, is_in=False)
        X_target.extend(feats)
        Y_target.extend([0]*bs)

    X_target = np.array(X_target)
    Y_target = np.array(Y_target)

    y_pred = attack_clf.predict(X_target)
    return compute_metrics(y_pred, Y_target)
