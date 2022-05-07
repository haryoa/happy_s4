from torch.nn.modules import Module
from torch.nn import Embedding
import torch.nn.functional as F
from happy_s4.model.s4 import S4
from dataclasses import dataclass
from torch.nn import LSTM, Linear
import torch


@dataclass
class LSTM_ARGS:
    pad_token_id: int
    vocab_size: int
    d_model: int
    channels: int
    bidirectional: bool = True
    pool: str = "last"
    n_layers: int = 1
    num_labels: int = 2
    dropout: float = 0.00


class LSTM_BRR(Module):
    def __init__(self, args: LSTM_ARGS):
        super().__init__()
        self.args = args
        d_model = args.d_model
        self.lstm = LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
        self.embedding = Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.d_model,
            padding_idx=args.pad_token_id,
        )

    def forward(self, input_ids, attention_masks):
        lstm_out = self.embedding(input_ids)
        lstm_out = self.lstm(lstm_out)

        if self.args.pool == "last":
            len_batch = attention_masks.sum(dim=1)
            pooled_out = lstm_out[torch.arange(lstm_out.size(0)), len_batch - 1, :]
        return pooled_out


class LSTM_GO_BRR_Classification(Module):
    def __init__(self, args: LSTM_ARGS):
        super().__init__()
        self.backbone = LSTM_BRR(args)
        self.out_linear = Linear(args.d_model, args.num_labels)
        self.args = args

    def forward(self, input_ids, attention_masks, labels=None):
        out_backbone = self.backbone(input_ids, attention_masks)
        logits = self.out_linear(out_backbone)
        loss = F.cross_entropy(logits, labels)
        return loss, out_backbone
