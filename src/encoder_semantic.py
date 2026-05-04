"""Semantic encoder wrapper around a HuggingFace transformer

Mean-pools the last hidden state over the attention mask and L2-normalizes
Loads from a local snapshot only (air-gap-friendly)
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SemanticEncoder:
    def __init__(self, local_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(local_path, local_files_only=True).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64, max_length: int = 64) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu())
        return torch.cat(out, dim=0)
