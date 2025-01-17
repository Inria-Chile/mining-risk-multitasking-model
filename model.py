from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning.metrics.classification import F1, Precision, Recall
from pytorch_lightning.metrics.regression import MSE

from sklearn.metrics import ndcg_score

import pandas as pd
import numpy as np

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": F.relu,
}


class MultiTaskLearner(LightningModule):
    def __init__(
        self,
        regression_task,
        classification_task,
        input_size,
        hidden_size,
        learning_rate,
        classifier_lambda,
        tanh_loss,
        fill_missing_regression,
        regressor_activation,
        classifier_loss_weights=None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        # Sanity checks
        assert 0 <= classifier_lambda <= 1

        # Layers
        self.hidden_fc = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias=True
        )
        
        if classification_task:
            self.classifier_hidden_fc = nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True
            )
            self.classifier_fc = nn.Linear(
                in_features=hidden_size,
                out_features=2,  # It's a binary classification
                bias=True,
            )
            self._build_classifier_loss_weigths(classifier_loss_weights)

        if regression_task:
            self.regressor_hidden_fc = nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
            )
            self.regressor_fc = nn.Linear(
                in_features=hidden_size, out_features=1, bias=True  # It's a regression
            )

        # Metrics
        if classification_task:
            self.f1 = F1()
            self.precision= Precision()
            self.recall = Recall()
        if regression_task:
            self.mse = MSE()
    
    def _build_classifier_loss_weigths(self, classes_count):
        normed_count = [1 - (x / sum(classes_count)) for x in classes_count]
        weights_tensor = torch.tensor(normed_count, dtype=torch.float)
        self.register_buffer("_classifier_loss_weights", weights_tensor)
        
    def forward(self, features):
        hidden_features = self.hidden_fc(features)
        hidden_features = torch.tanh(hidden_features)
        if self.hparams.classification_task:
            classification_hidden = self.classifier_hidden_fc(hidden_features)
            classification_hidden = torch.tanh(classification_hidden)
            classification = self.classifier_fc(classification_hidden)
        else:
            classification = None
        
        if self.hparams.regression_task:
            regression_hidden = self.regressor_hidden_fc(hidden_features)
            regression_hidden = ACTIVATIONS[self.hparams.regressor_activation](regression_hidden)
            regression = self.regressor_fc(regression_hidden)
        else:
            regression = None
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
        if self.hparams.classification_task:
            classification_loss = F.cross_entropy(
                input=classification_predicted,
                target=classification_target,
                weight=self._classifier_loss_weights,
                reduction="mean"
            )
        else:
            classification_loss = torch.zeros(1).to(self.device)

        if self.hparams.regression_task:
            regression_mask = torch.isnan(regression_target)
            regression_predicted = regression_predicted.squeeze()
            regression_predicted[regression_mask] = 0
            regression_target[regression_mask] = 0
            regression_loss = F.mse_loss(
                input=regression_predicted,
                target=regression_target,
                reduction="mean"
            )
        else:
            regression_loss = torch.zeros(1).to(self.device)
        
        if self.hparams.tanh_loss:
            classification_loss = torch.tanh(classification_loss)
            regression_loss = torch.tanh(regression_loss)

        if self.hparams.classifier_lambda > 0:
            lambda_ = self.hparams.classifier_lambda
            loss = lambda_ * classification_loss + (1 - lambda_) * regression_loss
        else:
            loss = classification_loss + regression_loss

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

        years = batch["year"].tolist()
        months = batch["month"].tolist()
        regions = batch["region"]
        relevances = batch["relevance"].tolist()

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
            "years": years,
            "months": months,
            "regions": regions,
            "relevances": relevances,
        }

    def _epoch_end_metrics(self, outputs, prefix=""):
        # Loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Classification metrics
        if self.hparams.classification_task:
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
        else:
            classification_metrics = {}

        # Regression metrics
        if self.hparams.regression_task:
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
        else:
            regression_metrics = {}
        
        # Ranking metrics
        ranking_metrics = self._ranking_metrics(outputs, prefix)

        return avg_loss, classification_metrics, regression_metrics, ranking_metrics

    @staticmethod
    def _mask_regressor_tensors(target, predicted):
        mask = ~torch.isfinite(target)
        predicted[mask] = 0
        target[mask] = 0
        return target, predicted

    def training_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def training_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics, ranking_metrics = self._epoch_end_metrics(outputs, prefix="train")
        return {"loss": loss, "log": {"train_loss": loss, **classifier_metrics, **regressor_metrics, **ranking_metrics}}

    def validation_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics, ranking_metrics = self._epoch_end_metrics(
            outputs, prefix="val"
        )
        return {"val_loss": loss, "log": {"val_loss": loss, **classifier_metrics, **regressor_metrics, **ranking_metrics}}

    def test_step(self, batch, batch_idx):
        return self._inference(batch, batch_idx)

    def test_epoch_end(self, outputs):
        loss, classifier_metrics, regressor_metrics, ranking_metrics = self._epoch_end_metrics(
            outputs, prefix="test"
        )
        return {"test_loss": loss, "log": {"test_loss": loss, **classifier_metrics, **regressor_metrics, **ranking_metrics}}
    
    def _ranking_metrics(self, outputs, prefix):
        years = [x for out in outputs for x in out["years"]]
        months = [x for out in outputs for x in out["months"]]
        regions = [x for out in outputs for x in out["regions"]]
        relevances = [x for out in outputs for x in out["relevances"]]

        if self.hparams.classification_task:
            criticality = torch.cat(
                [x["classifier_tensors"][1] for x in outputs]
            ).softmax(dim=1)[:,1].tolist()
        else:
            criticality = (-1 * torch.cat([x["regressor_tensors"][1] for x in outputs]).squeeze()).tolist()


        df = pd.DataFrame({
            "year": years,
            "month": months,
            "region": regions,
            "relevance": relevances,
            "criticality": criticality,
        })

        ndcgs = defaultdict(list)
        grouped_by_df = df.groupby(["region", "year", "month"])
        missing = 0
        for region_year_month, group_df in grouped_by_df:
            if group_df["relevance"].sum() == 0:
                missing += 1
                continue
            if len(group_df) > 1:
                score = ndcg_score(
                    y_true=[group_df["relevance"]],
                    y_score=[group_df["criticality"]],
                )
            else:
                score = 1
            region = region_year_month[0]
            ndcgs[region].append(score)
        
        mean_ndcgs = {}
        for key, value in ndcgs.items():
            mean_key = f"{prefix}_{key}_NDCG"
            mean_value = np.mean(value)
            mean_ndcgs[mean_key] = mean_value
        
        mean_ndcg = np.mean(list(mean_ndcgs.values()))
        mean_ndcgs[f"{prefix}_NDCG"] = mean_ndcg
        
        return mean_ndcgs


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--regression_task", type=bool, default=False)
        parser.add_argument("--classification_task", type=bool, default=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--classifier_lambda", type=float, default=0)
        parser.add_argument("--input_size", type=int, default=24)
        parser.add_argument("--hidden_size", type=int, default=50)
        parser.add_argument("--tanh_loss", type=bool, default=False)
        parser.add_argument("--regressor_activation", type=str, default="tanh")

        return parser
