import sys
import traceback
sys.path.append(".")

from happy_s4.experiment import Experiment, ExperimentArgs
from pytorch_lightning import seed_everything
from happy_s4.database.data import DATA_INP_LIST

trainer_args = {
    "gpus": [0],
    "max_epochs": 1000,
}


model_args = {
    "d_model": 128,
    "l_max": 4000,
    "channels": 1,
    "hurwitz": True,
    "transposed": False,
    "dropout": 0.0,
    "n_layers": 4,
    "trainable": {"dt": True, "A": True, "P": True, "B": True},
    "bidirectional": True,
}


lit_model_args = {
    "learning_rate" : 1e-3,
    "model_type": "s4-clf"
}

# SEE HFLITDATAARGS
batch_args = {
    "total_batch_size": 64,
    "grad_accumulation": 1,
}
data_args = {
    "data": "superglue/rte",
    "args": {
        "split_pattern": "\s+",
    },
}


early_stopping_args = dict(monitor="val_loss", patience=5)
model_checkpoint_args = dict(
    filename="checkpoint_{epoch:02d}-{val_loss:.4f}", monitor="val_loss"
)


model_name = "s4-exp-word-1"


def main():
    seed_everything(1234, workers=True)
    for data in DATA_INP_LIST:
        try:
            print(f"{data} train")
            out_dir = f"outputs/{model_name}/{data}"
            data_args["data"] = data
            args = ExperimentArgs(
                out_dir,
                data_args,
                batch_args,
                trainer_args,
                model_args,
                early_stopping_args,
                model_checkpoint_args,
                lit_model_args=lit_model_args
            )
            
            exp = Experiment(args)
            exp.run()
        except:
            print(traceback.print_exc())
            print(f"Failed to train {data}! Skipping~")


if __name__ == "__main__":
    main()
