# -*- coding: utf-8 -*-
"""
tokenizer.py
职责：
1) 加载 HuggingFace BERT tokenizer + BERT encoder
2) 从文本提取 768 维 CLS 向量（last_hidden_state[:,0,:]）
"""

from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Bert768Encoder:
    """
    用 BERT 输出 768 维 CLS 向量。
    默认：bert-base-chinese（hidden_size=768）
    """
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.py 里 Bert768Encoder.__init__ 加上
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.bert.to(self.device)  # <- 新增
        self.bert.eval()



    @torch.no_grad()
    def encode_768(
        self,
        texts: List[str],
        max_len: int = 256,
        batch_size: int = 32,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        输入：texts (List[str])
        输出：features (N, 768) float32

        normalize=True 时做 L2 normalize（有些下游检索/线性模型会更稳定）
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)

            out = self.bert(input_ids=input_ids, attention_mask=attn)
            cls = out.last_hidden_state[:, 0, :]  # [B, 768]

            if normalize:
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)

            feats.append(cls.cpu().numpy())

        arr = np.vstack(feats).astype(np.float32)
        # 安全检查
        if arr.shape[1] != 768:
            raise ValueError(f"Expected 768 dims, got {arr.shape}")
        return arr
