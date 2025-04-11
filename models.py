import torch
import torch.nn as nn
import torchvision.models as models

###############################################################################
# 1) DistilBERT model for text (e.g., SST-2)
###############################################################################
class DistilBertClassifier(nn.Module):
    """
    A wrapper around DistilBERT for sequence classification.
    Expects batches to include 'input_ids', 'attention_mask', and labels.
    """
    def __init__(self, num_classes=2):
        super(DistilBertClassifier, self).__init__()
        from transformers import DistilBertModel
        # Load DistilBert backbone
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Classifier head (DistilBert hidden_size is typically 768)
        hidden_size = self.distilbert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Grab the [CLS]-token embedding
        cls_hidden_state = outputs.last_hidden_state[:, 0]  
        logits = self.classifier(cls_hidden_state)
        return logits


###############################################################################
# Main get_model function
###############################################################################
def get_model(model_name, num_classes):
    """
    Returns a PyTorch model given the model_name and desired num_classes.
    model_name can be:
     - 'alexnet'
     - 'distilbert'  (for SST-2)
    """
    model_name = model_name.lower()
    
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        return model
    
    elif model_name == 'distilbert':
        # For SST-2 or other 2-class tasks, pass num_classes=2.
        return DistilBertClassifier(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
