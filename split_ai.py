import torch
import random
import os
from torch.utils.data import DataLoader, Subset
import copy

def move_to_device_and_cast(x, device='cuda'):
    """
    Function to handle both dict-based and tensor-based inputs.
    Ensures input_ids is cast to long, and everything is on 'device'.
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            tmp = v.to(device)
            if k == "input_ids":
                tmp = tmp.long()
            out[k] = tmp
        return out
    else:
        return x.to(device)

class SplitAIEnsemble:
    """
    Implements the 'Split-AI' ensemble architecture:
      - K sub-models
      - Each data sample is excluded from L sub-models
      - At inference for a known member sample, pick sub-models that did NOT see it
      - At inference for a non-member sample, pick L sub-models at random
    """
    def __init__(self, model_fn, K=25, L=10, device='cuda'):
        self.model_fn = model_fn
        self.K = K
        self.L = L
        self.device = device

        # Create K sub-models
        self.submodels = [model_fn().to(device) for _ in range(K)]
        
        # For each training sample i, store which sub-models do NOT see it:
        self.non_model_indices = {}
        
        # Also store sub-model -> which data indices it sees
        self.model_data_indices = [set() for _ in range(K)]

    def partition_data(self, dataset):
        """
        For each sample i, randomly choose L sub-models that do NOT see it.
        Then add i to the other (K - L) sub-models' subsets.
        """
        n = len(dataset)
        for i in range(n):
            non_models_for_i = random.sample(range(self.K), self.L)
            self.non_model_indices[i] = non_models_for_i
            for sub_idx in range(self.K):
                if sub_idx not in non_models_for_i:
                    self.model_data_indices[sub_idx].add(i)

    def build_subdatasets(self, dataset):
        """
        Creates a list of Subset objects, one for each sub-model's training subset.
        """
        subdatasets = []
        for sub_idx in range(self.K):
            indices_list = list(self.model_data_indices[sub_idx])
            subdatasets.append(Subset(dataset, indices_list))
        return subdatasets

    def train_submodels(self, dataset, batch_size, epochs, optimizer_fn, train_one_epoch_fn, num_workers=2):
        """
        Train each sub-model on its own subset of the data (single-threaded).
        """
        subdatasets = self.build_subdatasets(dataset)

        for sub_idx in range(self.K):
            print(f"[SplitAI] Training sub-model {sub_idx+1}/{self.K}...")
            loader = DataLoader(
                subdatasets[sub_idx],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            optimizer = optimizer_fn(self.submodels[sub_idx].parameters())
            
            for ep in range(epochs):
                avg_loss, avg_acc = train_one_epoch_fn(
                    self.submodels[sub_idx],
                    loader,
                    optimizer,
                    device=self.device
                )
                if (ep+1) % 2 == 0:
                    print(f"   [Sub-model {sub_idx}, epoch {ep+1}] loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    def save_submodels(self, folder_path, dataset_name="cifar10"):
        """
        Save each sub-model to disk with the format: splitai_submodel_{sub_idx}_{dataset_name}.pt
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sub_idx, model in enumerate(self.submodels):
            filename = f"splitai_submodel_{sub_idx}_{dataset_name}.pt"
            path = os.path.join(folder_path, filename)
            torch.save(model.state_dict(), path)
            print(f"[SplitAI] Saved sub-model {sub_idx} => {path}")
    
    def load_submodels(self, folder_path, dataset_name="cifar10"):
        """
        Load each sub-model from disk with the format: splitai_submodel_{sub_idx}_{dataset_name}.pt
        """
        for sub_idx in range(self.K):
            filename = f"splitai_submodel_{sub_idx}_{dataset_name}.pt"
            path = os.path.join(folder_path, filename)
            if os.path.exists(path):
                self.submodels[sub_idx].load_state_dict(torch.load(path))
                print(f"[SplitAI] Loaded sub-model {sub_idx} => {path}")
            else:
                print(f"[SplitAI] WARNING: no checkpoint found for sub-model {sub_idx} at {path}")

    def _forward_submodel(self, sub_idx, x_dev):
        """
        If x_dev is a dict, call self.submodels[sub_idx](**x_dev).
        Otherwise call self.submodels[sub_idx](x_dev).
        """
        if isinstance(x_dev, dict):
            return self.submodels[sub_idx](**x_dev)
        else:
            return self.submodels[sub_idx](x_dev)

    def inference_member(self, x, x_idx):
        """
        For a known member sample x at index x_idx:
        We only use sub-models that did NOT see x. Then average their outputs.
        """
        not_seen_indices = self.non_model_indices[x_idx]
        outputs = []
        with torch.no_grad():
            for sub_idx in not_seen_indices:
                x_dev = move_to_device_and_cast(x, self.device)
                logit = self._forward_submodel(sub_idx, x_dev)
                outputs.append(logit)
        return torch.mean(torch.stack(outputs), dim=0)

    def inference_nonmember(self, x):
        """
        For a non-member sample, randomly pick L sub-models from [0..K-1], then average.
        """
        not_seen_indices = random.sample(range(self.K), self.L)
        outputs = []
        with torch.no_grad():
            for sub_idx in not_seen_indices:
                x_dev = move_to_device_and_cast(x, self.device)
                logit = self._forward_submodel(sub_idx, x_dev)
                outputs.append(logit)
        return torch.mean(torch.stack(outputs), dim=0)
