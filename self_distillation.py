import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class DistilledDataset(torch.utils.data.Dataset):
    """
    A custom dataset that holds distilled data as (input, soft_label) pairs.
    soft_label is a probability/confidence distribution (or logits) from an ensemble.
    """
    def __init__(self, X_list, Y_soft):
        super().__init__()
        self.X_list = X_list
        self.Y_soft = Y_soft

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.Y_soft[idx]

def get_soft_labels(split_ai_ensemble, train_dataset, device='cuda', batch_size=64):
    """
    Queries the SplitAIEnsemble on every training sample once to build soft labels.
    
    Returns:
      X_list: a list of input samples (CPU tensors or dictionaries)
      Y_soft: a list of corresponding soft label logits (CPU tensors)
    """
    X_list = []
    Y_soft = []
    
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    start_idx = 0
    for batch in tqdm(data_loader, desc="Creating Soft Labels"):
        # If the batch is a dict (e.g. SST-2), convert
        if isinstance(batch, dict):
            batch_inputs = {
                "input_ids": batch["input_ids"].to(device).long(),
                "attention_mask": batch["attention_mask"].to(device)
            }
            batch_labels = batch["label"].to(device).long() if isinstance(batch["label"], torch.Tensor) \
                          else torch.tensor(batch["label"], dtype=torch.long, device=device)
        else:
            # For images or FC data
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
        
        # Determine batch size
        if isinstance(batch_labels, torch.Tensor):
            bs = batch_labels.size(0)
        else:
            bs = len(batch_labels)
        
        for i in range(bs):
            idx_in_dataset = start_idx + i
            # Extract the i-th sample
            if isinstance(batch_inputs, dict):
                single_input = {k: v[i].unsqueeze(0) for k, v in batch_inputs.items()}
            else:
                single_input = batch_inputs[i].unsqueeze(0)
            # Query sub-models that didn't see this sample
            with torch.no_grad():
                logit = split_ai_ensemble.inference_member(single_input, idx_in_dataset)
            # Store the input on CPU
            if isinstance(batch_inputs, dict):
                X_list.append({k: v[i].cpu() for k, v in batch_inputs.items()})
            else:
                X_list.append(batch_inputs[i].cpu())
            Y_soft.append(logit.cpu().squeeze(0))
        start_idx += bs
    return X_list, Y_soft

def train_distilled_model(model, distilled_dataset, epochs=10, batch_size=64, device='cuda', lr=0.001):
    """
    Trains the final single distilled model using the DistilledDataset.
    The soft labels (teacher outputs) are interpreted as distributions and
    the KLDivLoss is used for training.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    loader = DataLoader(distilled_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        total_samples = 0
        for X_soft, Y_soft in loader:
            # Move everything to device
            if isinstance(X_soft, dict):
                X_soft = {k: v.to(device) for k, v in X_soft.items()}
            else:
                X_soft = X_soft.to(device)
            Y_soft = Y_soft.to(device)

            optimizer.zero_grad()
            teacher_probs = torch.softmax(Y_soft, dim=1)
            if isinstance(X_soft, dict):
                student_logits = model(**X_soft)
            else:
                student_logits = model(X_soft)
            student_log_probs = torch.log_softmax(student_logits, dim=1)
            loss = loss_fn(student_log_probs, teacher_probs)
            loss.backward()
            optimizer.step()

            batch_size_now = Y_soft.size(0)
            epoch_loss += loss.item() * batch_size_now
            total_samples += batch_size_now
        
        avg_epoch_loss = epoch_loss / total_samples
        print(f"[Self-Distillation] epoch={ep+1}, loss={avg_epoch_loss:.4f}")
    return model
