from functools import partial

from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
from typing import List, Optional
import datasets
from torch.utils.data import DataLoader

from happy_s4.data.data import WordTokenizer, shape_dataset
from happy_s4.data.collators import BatchCollators


@dataclass
class HFLitDataArgs:
    min_freq: int
    split_pattern: str = r"\s+"
    data_name: str = "glue"
    data_subset: str = "mrpc"
    data_split_train: str = "train"
    data_split_val: str = "validation"
    data_split_test: str = "test"
    batch_size: int = 32
    column_train: Optional[List[str]] = None


class HFLitDataModule(LightningDataModule):
    def __init__(self, args: HFLitDataArgs):
        super().__init__()
        self.args = args
        self.data_train = None
        self.data_val = None
        self.tokenizer = None
        if self.args.column_train is None:
            self.args.column_train = ["sentence1", "sentence2"]

    def prepare_data(self) -> None:
        datasets.load_dataset(
            self.args.data_name,
            name=self.args.data_subset,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.data_train = datasets.load_dataset(
                self.args.data_name,
                name=self.args.data_subset,
                split=self.args.data_split_train,
            )
            self.tokenizer = WordTokenizer(
                lowercase=True, special_tokens=["[sep]", "[cls]"]
            )
            args_input = [self.data_train[x] for x in self.args.column_train]
            self.tokenizer.fit(*args_input)
            older_column = self.data_train.column_names
            older_column.pop("label")
            self.data_train = self.data_train.map(
                partial(shape_dataset, word_tokenizer=self.tokenizer)
            )
            self.data_train = self.data_train.remove_columns(older_column)

            self.data_val = datasets.load_dataset(
                self.args.data_name,
                name=self.args.data_subset,
                split=self.args.data_split_train,
            )
            self.data_val = self.data_val.map(
                partial(shape_dataset, word_tokenizer=self.tokenizer)
            )
            self.data_val = self.data_val.remove_columns(older_column)

    def train_dataloader(self):
        dl = DataLoader(
            dataset=self.data_train,
            collate_fn=BatchCollators(self.tokenizer.pad_token_id),
            batch_size=self.args.batch_size,
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            dataset=self.data_val,
            collate_fn=BatchCollators(self.tokenizer.pad_token_id),
            batch_size=self.args.batch_size,
        )
        return dl

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
