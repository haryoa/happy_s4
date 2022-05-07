from torch.nn.modules import Module
from torch.nn import Embedding
import torch.nn.functional as F
from happy_s4.model.s4 import S4
from dataclasses import dataclass
from torch.nn import ModuleList
import torch


@dataclass
class S4_GO_BRR_ARGS:
    pad_token_id: int
    vocab_size: int
    d_model: int
    l_max: int
    channels: int
    bidirectional: bool = True
    trainable: bool = True
    lr: float = 0.001
    tie_state: bool = True
    hurwitz: bool = True
    transposed: bool = True
    pool: str = "last"
    n_layers: int = 1
    dropout: float = 0.00

    def get_s4_args(self):
        args = [
            "d_model",
            "l_max",
            "channels",
            "bidirectional",
            "trainable",
            "lr",
            "tie_state",
            "hurwitz",
            "transposed",
            "dropout"
        ]
        dict_returned = {arg: self.__dict__[arg] for arg in args}
        return dict_returned


class S4_GO_BRR(Module):
    def __init__(self, args: S4_GO_BRR_ARGS):
        super().__init__()
        self.args = args
        if self.args.n_layers > 1:
            self.s4_s = ModuleList([S4(**args.get_s4_args()) for i in range(self.args.n_layers)])
        else:
            self.s4 = S4(**args.get_s4_args())
        self.args = args
        self.embedding = Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.d_model,
            padding_idx=args.pad_token_id,
        )

    def forward(self, input_ids, attention_masks):
        s4_out = self.embedding(input_ids)
        if self.args.n_layers > 1:
            for s4_model in self.s4_s:
                s4_out, hdn = s4_model(s4_out)
        else:
            s4_out, hdn = self.s4(s4_out)
        if self.args.pool == "last":
            len_batch = attention_masks.sum(dim=1)
            pooled_out = s4_out[torch.arange(s4_out.size(0)), len_batch - 1, :]
        return pooled_out


class S4_GO_BRR_Classification(Module):
    def __init__(self, args: S4_GO_BRR_ARGS):
        super().__init__()
        self.backbone = S4_GO_BRR(args)
        self.args = args

    def forward(self, input_ids, attention_masks, labels=None):
        s4_out = self.backbone(input_ids, attention_masks)
        loss = F.cross_entropy(s4_out, labels)
        return loss, s4_out
