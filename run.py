from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from dataset import WorksitesDataset
from model import MultiTaskLearner

HYPERPARAMETERS = {
    "batch_size": 32,
    "input_size": 24,
    "hidden_size": 50,
    "learning_rate": 0.0005,
    "epochs": 150,
    "multitasking_enabled": True,
    "lr_scheduler_enabled": False,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 5,
    "lr_scheduler_verbose": True,
}

LABEL_COLUMNS = ["FUTURE_TOTAL_COUNT", "DAYS_UNTIL_NEXT_ACCIDENT"]
BASE_PATH = Path(".")
DATASET_PATH = BASE_PATH / "datasets/worksites.csv"


def main(_):
    # Build datasets
    train_ds = WorksitesDataset(
        csv_path=DATASET_PATH, split=WorksitesDataset.TRAIN, label_columns=LABEL_COLUMNS
    )
    val_ds = WorksitesDataset(
        csv_path=DATASET_PATH,
        split=WorksitesDataset.VAL,
        label_columns=LABEL_COLUMNS,
        feature_scaler=train_ds._feature_scaler,
    )
    test_ds = WorksitesDataset(
        csv_path=DATASET_PATH,
        split=WorksitesDataset.TEST,
        label_columns=LABEL_COLUMNS,
        feature_scaler=train_ds._feature_scaler,
    )

    # Build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
    )
    val_dl = DataLoader(val_ds, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)
    test_dl = DataLoader(
        test_ds, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False
    )

    # Instantiate model and train
    model = MultiTaskLearner(
        input_size=HYPERPARAMETERS["input_size"],
        hidden_size=HYPERPARAMETERS["hidden_size"],
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, train_dl)


if __name__ == "__main__":
    main([])
