import torch
import torch.nn.functional as F

def train_one_epoch_selena(model, train_loader, optimizer, device='cuda'):
    """
    Sub-model training for SELENA's Split-AI.
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for batch in train_loader:
        if isinstance(batch, dict):
            # DistilBERT style
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc

def eval_model_selena(model, test_loader, device='cuda'):
    model.eval()
    total_correct, total_samples = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids, attention_mask=attention_mask)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = model(images)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples
