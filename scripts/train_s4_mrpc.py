import sys

sys.path.append(".")

from happy_s4.data.lit_data import HFLitDataArgs, HFLitDataModule
from happy_s4.model.lit_model import LitS4Args, LitS4
from happy_s4.model.s4_wrap import S4_GO_BRR_ARGS
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from happy_s4.data.data import WordTokenizer, shape_dataset
from happy_s4.data.collators import BatchCollators
import datasets
from functools import partial
from torch.utils.data import DataLoader
import datasets
from functools import partial
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def main():
    lit_args = HFLitDataArgs(min_freq=1)
    dm = HFLitDataModule(lit_args)
    dm.setup()
    vocab_size = len(dm.tokenizer.vocab.get_itos())
    pad_token_id = dm.tokenizer.pad_token_id
    s4_args = S4_GO_BRR_ARGS(pad_token_id=pad_token_id, vocab_size=vocab_size, d_model=256, l_max=2000, channels=1,hurwitz=True, transposed=False)
    lit_s4_args = LitS4Args(s4_args=s4_args)
    lit_s4 = LitS4(lit_s4_args)
    es = EarlyStopping(monitor="val_loss")
    mckpt = ModelCheckpoint(dirpath="outputs/mrpc/", filename="checkpoint_{epoch:02d}-{val_loss:03.0f}", monitor='val_loss', )
    trainer = Trainer(callbacks=[es, mckpt], gpus=[1], max_epochs=1000)
    trainer.fit(model=lit_s4, datamodule=dm)
    
    lit_s4 = lit_s4.load_from_checkpoint("outputs/mrpc/checkpoint_epoch=03-val_loss=001.ckpt")
        data_test = datasets.load_dataset(
        "glue",
        "mrpc",
        split="test",
    )
    data_test = data_test.map(
        partial(shape_dataset, word_tokenizer=dm.tokenizer)
    )
    data_test = data_test.remove_columns(['sentence1', 'sentence2', 'idx'])
    dl = DataLoader(
        dataset=data_test,
        collate_fn=BatchCollators(dm.tokenizer.pad_token_id),
        batch_size=dm.args.batch_size,
        shuffle=False
    )
    preds = []
    gt = []
    
    for batch in tqdm(dl):
    with torch.no_grad(): 
        _, logits = lit_s4(input_ids=batch['input_ids'], labels=batch['label'])
        label = logits.argmax(dim=1).numpy().tolist()
        gt.extend(batch['label'].numpy().tolist())
        preds.extend(label)
    print(accuracy_score(gt, preds))
    


if __name__ == "__main__":
    main()