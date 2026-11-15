from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


class Splade:
    def __init__(self, model_dir=None, device="cpu"):
        model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "model"
        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_config(config)
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    @torch.inference_mode()
    def encode(self, texts, max_length=256, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 2 <= max_length <= self.model.config.max_position_embeddings:
            raise ValueError("max_length must be between 2 and 512")
        if not texts:
            return torch.empty((0, self.model.config.vocab_size))
        vectors = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits
            weights = torch.log1p(torch.relu(logits))
            weights = weights * inputs["attention_mask"].unsqueeze(-1)
            vectors.append(weights.max(dim=1).values.cpu())
        return torch.cat(vectors)

    def encode_queries(self, texts, batch_size=8):
        return self.encode(texts, max_length=256, batch_size=batch_size)

    def encode_documents(self, texts, batch_size=8):
        return self.encode(texts, max_length=256, batch_size=batch_size)
