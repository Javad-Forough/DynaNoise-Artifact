import torch
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, device='cuda'):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for batch in train_loader:
        # Check if we're dealing with DistilBERT-style batches (dict) or image/tensor style (tuple)
        if isinstance(batch, dict):
            # Typical HuggingFace batch has input_ids, attention_mask, label
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            # Otherwise, assume tuple (images, labels)
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
    acc = total_correct / total_samples
    return avg_loss, acc

def eval_model(model, test_loader, device='cuda'):
    model.eval()
    total_correct, total_samples = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                # DistilBERT batch
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
    
    acc = total_correct / total_samples
    return acc

def eval_model_with_noise(model, test_loader, dyna_noise, device='cuda'):
    """
    Evaluate the model on test_loader, applying DynaNoise to the logits 
    before predicting. Returns accuracy with defense applied.
    """
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
            
            probs = dyna_noise.forward(logits)  # apply noise + smoothing
            preds = probs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    acc = total_correct / total_samples
    return acc
