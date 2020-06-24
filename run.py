from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from dataset import WorksitesDataset
from model import MultiTaskLearner

LABEL_COLUMNS = ["FUTURE_TOTAL_COUNT", "DAYS_UNTIL_NEXT_ACCIDENT"]


def main(_):
    # Build datasets
    train_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        split=WorksitesDataset.TRAIN,
        label_columns=LABEL_COLUMNS,
    )
    val_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        split=WorksitesDataset.VAL,
        label_columns=LABEL_COLUMNS,
        feature_scaler=train_ds._feature_scaler,
    )
    test_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        split=WorksitesDataset.TEST,
        label_columns=LABEL_COLUMNS,
        feature_scaler=train_ds._feature_scaler,
    )

    # Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate model and train
    dict_args = vars(args)
    model = MultiTaskLearner(**dict_args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dl)


if __name__ == "__main__":

    parser = ArgumentParser()

    # Program specific arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dataset_path", type=str, default="./datasets/worksites.csv")

    # Model specific arguments
    parser = MultiTaskLearner.add_model_specific_args(parser)

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
