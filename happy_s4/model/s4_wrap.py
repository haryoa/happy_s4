from torch.nn.modules import Module
from torch.nn import Embedding
import torch.nn.functional as F
from happy_s4.model.s4 import S4
from dataclasses import dataclass


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
    hurwitz: bool = (True,)
    transposed: bool = True
    pool: str = "last"

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
        ]
        dict_returned = {arg: self.__dict__[arg] for arg in args}
        return dict_returned


class S4_GO_BRR(Module):
    def __init__(self, args: S4_GO_BRR_ARGS):
        super().__init__()
        self.s4 = S4(**args.get_s4_args())
        self.args = args
        self.embedding = Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.d_model,
            padding_idx=args.pad_token_id,
        )

    def forward(self, input_ids):
        s4_out = self.embedding(input_ids)
        forward_s4 = self.s4(s4_out)
        if self.args.pool == "last":
            pooled_out = forward_s4[0][:, -1]
        return pooled_out


class S4_GO_BRR_Classification(Module):
    def __init__(self, args: S4_GO_BRR_ARGS):
        super().__init__()
        self.backbone = S4_GO_BRR(args)
        self.args = args

    def forward(self, input_ids, labels):
        s4_out = self.backbone(input_ids)
        loss = F.cross_entropy(s4_out, labels)
        return loss, s4_out
