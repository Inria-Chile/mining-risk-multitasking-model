from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning.metrics.classification import F1, Precision, Recall
from pytorch_lightning.metrics.regression import MSE


class MultiTaskLearner(LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # Layers
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

        # Metrics
        self.f1 = F1()
        self.precision = Precision()
        self.recall = Recall()
        self.mse = MSE()

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
        num_targets = torch.tensor(classification_target.size()).to(self.device)
        negative_classes_count = (num_targets - positive_classes_count).float()
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

    # PyTorch Lightning hooks

    def _inference(self, batch, _):
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

        return {
            "loss": loss,
            "classifier_tensors": (classifier_target, classifier_predicted),
            "regressor_tensors": (regressor_target, regressor_predicted),
        }

    def _epoch_end_metrics(self, outputs, prefix=""):
        # Loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Classification metrics
        classification_targets = torch.cat(
            [x["classifier_tensors"][0] for x in outputs]
        )
        classification_predicted = torch.cat(
            [x["classifier_tensors"][1] for x in outputs]
        ).argmax(dim=1)
        classification_metrics = {
            f"{prefix}_f1": self.f1(classification_predicted, classification_targets),
            f"{prefix}_precision": self.precision(
                classification_predicted, classification_targets
            ),
            f"{prefix}_recall": self.recall(
                classification_predicted, classification_targets
            ),
        }

        # Regression metrics
        regression_targets = torch.cat([x["regressor_tensors"][0] for x in outputs])
        regression_predicted = torch.cat(
            [x["regressor_tensors"][1] for x in outputs]
        ).squeeze()
        regression_targets, regression_predicted = self._mask_regressor_tensors(
            regression_targets, regression_predicted
        )

        regression_metrics = {
            f"{prefix}_mse": self.mse(regression_targets, regression_predicted)
        }

        return avg_loss, classification_metrics, regression_metrics

    @staticmethod
    def _mask_regressor_tensors(target, predicted):
        mask = ~torch.isfinite(target)
        predicted[mask] = 0
        target[mask] = 0
        return target, predicted

    def training_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def training_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics = self._epoch_end_metrics(outputs)
        return {"loss": loss, **classifier_metrics, **regressor_metrics}

    def validation_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics = self._epoch_end_metrics(
            outputs, prefix="val"
        )
        return {"val_loss": loss, **classifier_metrics, **regressor_metrics}

    def test_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def test_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics = self._epoch_end_metrics(
            outputs, prefix="test"
        )
        return {"test_loss": loss, **classifier_metrics, **regressor_metrics}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_size", type=int, default=24)
        parser.add_argument("--hidden_size", type=float, default=50)

        return parser
