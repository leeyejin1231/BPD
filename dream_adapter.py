from types import SimpleNamespace

import torch
from transformers import AutoModel

MASK_ID = 151666
END_IDS = (151643, 151645)


class DreamForwardAdapter:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, input_ids, attention_mask=None, **unused):
        if attention_mask is not None:
            raise ValueError('DreamForwardAdapter expects attention_mask=None (unpadded batch of 1).')
        logits = self.model(input_ids, attention_mask='full', position_ids=None).logits
        return SimpleNamespace(logits=torch.cat([logits[:, :1], logits[:, :-1]], dim=1))


def load_model(path, device='cuda'):
    model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    return DreamForwardAdapter(model)
