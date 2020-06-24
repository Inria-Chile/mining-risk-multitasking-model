from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule


class MultiTaskLearner(LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.hidden_fc = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias=True
        )
        self.tanh = nn.Tanh()
        self.classifier_fc = nn.Linear(
            in_features=hidden_size,
            out_features=2,  # It's a binary classification
            bias=True,
        )
        self.regressor_fc = nn.Linear(
            in_features=hidden_size, out_features=1, bias=True  # It's a regression
        )

    def forward(self, features):
        hidden_features = self.hidden_fc(features)
        hidden_features = self.tanh(hidden_features)
        classification = self.classifier_fc(hidden_features)
        regression = self.regressor_fc(hidden_features)
        return classification, regression

    def loss(
        self,
        classification_predicted,
        regression_predicted,
        classification_target,
        regression_target,
    ):
        """
        Args:
            - classification_predicted: tensor with shape [batch_sz, 2]
            - regression_predicted: tensor with shape [batch_sz]
            - classification_target: tensor with shape [batch_sz]
            - regression_target: tensor with shape [batch_sz]
        Returns:
            - loss
            - loss_metrics
        """
        positive_classes_count = classification_target.sum().unsqueeze(0).float()
        negative_classes_count = (
            torch.tensor(classification_target.size()) - positive_classes_count
        ).float()
        classification_criterion = nn.CrossEntropyLoss(
            weight=torch.cat([positive_classes_count, negative_classes_count], dim=0)
        )
        regression_criterion = nn.MSELoss()
        classification_loss = classification_criterion(
            classification_predicted, classification_target
        )

        regression_mask = torch.isnan(regression_target)
        regression_predicted[regression_mask] = 0
        regression_target[regression_mask] = 0
        regression_loss = regression_criterion(
            regression_predicted.squeeze(), regression_target
        )

        loss = classification_loss + (
            regression_loss  # if HYPERPARAMETERS["multitasking_enabled"] else 0
        )
        return (
            loss,
            {
                "loss": loss.item(),
                "classification_loss": classification_loss.item(),
                "regression_loss": regression_loss.item(),
            },
        )

    @staticmethod
    def mask_regressor_tensors(target, predicted):
        mask = ~torch.isfinite(target)
        predicted[mask] = 0
        target[mask] = 0
        return target, predicted

    @staticmethod
    def classification_metrics(target, predicted):
        precision = precision_score(target, predicted)
        recall = recall_score(target, predicted)
        f1 = f1_score(target, predicted)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def regression_metrics(target, predicted):
        mse = mean_squared_error(target, predicted)
        return {
            "mse": mse,
        }

    # PyTorch Lightning methods

    def training_step(self, batch, _):
        # Unpack batch
        features = batch["features"]
        classifier_target = batch["classifier_target"]
        regressor_target = batch["regressor_target"]

        # Inference
        classifier_predicted, regressor_predicted = self(features)

        # Calculate loss, ignore loss metrics, and save them
        loss, _ = self.loss(
            classifier_predicted,
            regressor_predicted,
            classifier_target,
            regressor_target,
        )

        return {"loss": loss}

    def validation_step(self, batch, _):
        # Unpack batch
        features = batch["features"]
        classifier_target = batch["classifier_target"]
        regressor_target = batch["regressor_target"]

        # Inference
        classifier_predicted, regressor_predicted = self(features)

        # Calculate loss, ignore loss metrics, and save them
        loss, _ = self.loss(
            classifier_predicted,
            regressor_predicted,
            classifier_target,
            regressor_target,
        )

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def test_step(self, batch, _):
        # Unpack batch
        features = batch["features"]
        classifier_target = batch["classifier_target"]
        regressor_target = batch["regressor_target"]

        # Inference
        classifier_predicted, regressor_predicted = self(features)

        # Calculate loss, ignore loss metrics, and save them
        loss, _ = self.loss(
            classifier_predicted,
            regressor_predicted,
            classifier_target,
            regressor_target,
        )

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": avg_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_size", type=int, default=24)
        parser.add_argument("--hidden_size", type=float, default=50)

        return parser
