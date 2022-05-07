"""
Run Experiment Goes Here!
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, Optional
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rich.console import Console

from happy_s4.data.args import BatchArgs
from happy_s4.data.lit_data import HFLitDataArgs, HFLitDataModule
from happy_s4.database.data import DATA_INP_LIST
from happy_s4.model.lit_model import LitS4, LitS4Args
from happy_s4.model.s4_wrap import S4_GO_BRR_ARGS

console = Console()


@dataclass
class ExperimentArgs:
    """
    Experiment Arguments

    Parameters
    ----------
    out_dir: str
        Output directory of the model and other stuffs.
    data_args: Dict[str, Any]
        Data arguments, it should contains:
        ```
        data: what data you want to use e.g.: `super_glue/rte`
        args: HFLitDataArgs arguments (exclude batch size)
        ```
    batch_args: Dict[str, Any]
        Batch arguments, to compute the batch size. See BatchArgs
    trainer_args: Dict[str, Any]
        Trainer arguments for Pytorch Lightning Arguments.
        Please visit Pytorch Lightning Trainer's documentation
    model_args: Dict[str, Any]
        Model Arguments :3
    early_stopping_args: Dict[str, Any]
        Early Stopping Arguments in Pytorch Lightning Callback
    model_checkpoint_args: Dict[str, Any]
        Model checkpoint arguments in Pytorch LIghtning Calllback
    lit_model_args: Dict[str, Any]
        Lit model arguments
    """

    out_dir: str
    data_args: Dict[str, Any]
    batch_args: Dict[str, Any]
    trainer_args: Dict[str, Any]
    model_args: Dict[str, Any]
    early_stopping_args: Dict[str, Any]
    model_checkpoint_args: Dict[str, Any]
    lit_model_args: Optional[Dict[str, Any]] = None


class Experiment:
    """
    Experiment scripts
    """

    def __init__(self, args: ExperimentArgs) -> None:
        self.args = args
        console.log(args)

    def run(self) -> None:
        """
        Run the experiment
        """
        os.makedirs(self.args.out_dir, exist_ok=True)
        batch_args = BatchArgs(**self.args.batch_args)
        batch_size = int(
            batch_args.total_batch_size
            / batch_args.grad_accumulation
            / len(self.args.trainer_args["gpus"])
        )
        console.log(f"Batch size is [red]{batch_size}[/red]")
        lit_data_args = self.args.data_args.get("args")
        lit_data_args.update(DATA_INP_LIST.get(self.args.data_args.get("data"), {}))
        lit_data_args["batch_size"] = batch_size
        lit_data_args = HFLitDataArgs(**lit_data_args)
        dm = HFLitDataModule(lit_data_args)
        dm.setup()
        vocab_size = len(dm.tokenizer.vocab.get_itos())
        pad_token_id = dm.tokenizer.pad_token_id
        s4_args = S4_GO_BRR_ARGS(
            pad_token_id=pad_token_id, vocab_size=vocab_size, **self.args.model_args
        )

        # save tokenizer
        with open(Path(self.args.out_dir) / "tokenizer.pkl", "wb") as file:
            pickle.dump(dm.tokenizer, file)
        lit_model_args = {} if self.args.lit_model_args is None else self.args.lit_model_args
        lit_s4_args = LitS4Args(
            model_args=s4_args, **lit_model_args
        )
        lit_s4 = LitS4(lit_s4_args)
        es = EarlyStopping(**self.args.early_stopping_args)
        mckpt = ModelCheckpoint(
            dirpath=self.args.out_dir, **self.args.model_checkpoint_args
        )
        trainer = Trainer(
            callbacks=[es, mckpt],
            accumulate_grad_batches=batch_args.grad_accumulation,
            **self.args.trainer_args,
        )
        trainer.fit(model=lit_s4, datamodule=dm)
        with open(
            Path(self.args.out_dir) / "best_model_path.txt", "w+", encoding="utf-8"
        ) as file:
            file.write(mckpt.best_model_path)
            file.write(f" {mckpt.best_model_score.item()}")
