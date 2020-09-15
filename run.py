import os
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from dataset import WorksitesDataset
from model import MultiTaskLearner

LABEL_COLUMNS = ["FUTURE_TOTAL_COUNT", "DAYS_UNTIL_NEXT_ACCIDENT"]


def build_datasets(args, label_columns):
    train_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        fill_missing_regression=args.fill_missing_regression,
        split=WorksitesDataset.TRAIN,
        label_columns=label_columns,
    )
    val_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        fill_missing_regression=args.fill_missing_regression,
        split=WorksitesDataset.VAL,
        label_columns=label_columns,
        feature_scaler=train_ds._feature_scaler,
    )
    test_ds = WorksitesDataset(
        csv_path=args.dataset_path,
        fill_missing_regression=args.fill_missing_regression,
        split=WorksitesDataset.TEST,
        label_columns=label_columns,
        feature_scaler=train_ds._feature_scaler,
    )
    return train_ds, val_ds, test_ds


def main(args):
    # Sanity checks
    assert args.classification_task or args.regression_task

    seed_everything(42)

    # Build datasets
    train_ds, val_ds, test_ds = build_datasets(args, LABEL_COLUMNS)
    print("Size of train/val/test:", len(train_ds), len(val_ds), len(test_ds), end="\n\n")

    # Build dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
    )

    # Comet.ml logging
    if args.wandb_logging:
        wandb_logger = WandbLogger(
            name=args.wandb_name,
            project="mining"
        )

    # Instantiate model, train and test
    dict_args = vars(args)
    model = MultiTaskLearner(
        classifier_loss_weights=train_ds.classifier_weights,
        **dict_args
    )
    trainer = Trainer.from_argparse_args(
        args,
        default_root_dir=args.root_dir,
        early_stop_callback=False,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        logger=wandb_logger if args.wandb_logging else None,
    )
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)
    trainer.test(test_dataloaders=[test_dl])


if __name__ == "__main__":

    parser = ArgumentParser()

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # Program specific arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default="./datasets/worksites.csv")
    parser.add_argument("--wandb_logging", type=bool, default=False)
    parser.add_argument("--wandb_name", type=str, default="N/A")
    parser.add_argument("--dataloader_workers", type=int, default=8)
    parser.add_argument("--root_dir", type=str, default="logs/")
    parser.add_argument("--fill_missing_regression", type=int, default=-1)

    # Model specific arguments
    parser = MultiTaskLearner.add_model_specific_args(parser)

    args = parser.parse_args()
    print(args, end="\n\n")

    main(args)
