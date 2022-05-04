from re import X
from pytorch_lightning import LightningModule
from .s4_wrap import S4_GO_BRR_ARGS, S4_GO_BRR_Classification
from typing import List, Optional, Dict, Any, Union, Tuple
from torch.optim import Adam, AdamW
from dataclasses import dataclass


@dataclass
class LitS4Args:

    s4_args: S4_GO_BRR_ARGS
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    optimizer_beta: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001
    optimizer_epsilon: float = 1e-8


class LitS4(LightningModule):
    """
    Lightning Module for Seq2Sweq
    """

    def __init__(self, config: LitS4Args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self._init_model()

    def forward(self, **input_to_s4):  # type: ignore  # pylint: disable=all
        return self.model(**input_to_s4)

    def training_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all
        outputs = self(batch["input_ids"], batch["label"])
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all

        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        self.log("val_loss", val_loss)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):  # type: ignore  # pylint disable=all
        pass

    def test_step(self, batch, batch_idx):  # type: ignore  # pylint disable=all
        pass

    def test_epoch_end(self, outputs):  # type: ignore  # pylint disable=all
        pass

    def _init_model(self) -> None:
        if self.config.model_type == "s4-clf":
            self.model = S4_GO_BRR_Classification(self.config.s4_args)

    def configure_optimizers(self) -> Any:
        """
        Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self._get_optimizer(optimizer_grouped_parameters)

        return optimizer

    def _get_optimizer(
        self, optimizer_grouped_parameters: List[Dict[str, Any]]
    ) -> Union[Adam, AdamW]:
        if self.config.optimizer_type == "adamw":
            opt: Union[AdamW, Adam] = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.optimizer_beta,
                eps=self.config.optimizer_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adam":
            opt = Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.optimizer_beta,
                eps=self.config.optimizer_epsilon,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError("optimizer not implemented")
        return opt
