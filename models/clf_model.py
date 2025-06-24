import lightning as L
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers import AutoModel


class LitEarlyFusionModel(L.LightningModule):
    def __init__(
        self,
        num_classes,
        label_names,
        learning_rate=1e-5,
        feature_dim=1024,
        dropout_p=0.5,
        clinical_bert_name="emilyalsentzer/Bio_ClinicalBERT",
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Model Architecture (same as before) ---

        self.image_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_ftrs, self.hparams.feature_dim)
        self.report_model = AutoModel.from_pretrained(self.hparams.clinical_bert_name)
        self.text_projector = nn.Linear(
            self.report_model.config.hidden_size, self.hparams.feature_dim
        )
        self.clf_head = nn.Sequential(
            nn.Linear(self.hparams.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_p),
            nn.Linear(512, self.hparams.num_classes),
        )

        # --- Loss Function ---
        self.criterion = nn.BCEWithLogitsLoss()

        # --- Metrics ---
        # Use a ModuleDict to organize metrics for each phase
        self.metrics = nn.ModuleDict(
            {
                "train_auroc": MultilabelAUROC(num_labels=num_classes, average="macro"),
                "val_auroc": MultilabelAUROC(num_labels=num_classes, average="macro"),
                "test_auroc": MultilabelAUROC(num_labels=num_classes, average="macro"),
                "test_f1": MultilabelF1Score(num_labels=num_classes, average="macro"),
                "test_accuracy": MultilabelAccuracy(
                    num_labels=num_classes, average="macro"
                ),
            }
        )

        # To store outputs for the final report
        self.test_step_outputs = []

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
        text_outputs = self.report_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_token_output = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projector(cls_token_output)
        fused_features = torch.cat((image_features, text_features), dim=1)
        logits = self.clf_head(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        logits, loss, preds, labels = self._common_step(batch)
        self.metrics["train_auroc"].update(preds, labels.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_auroc", self.metrics["train_auroc"], on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, preds, labels = self._common_step(batch)
        self.metrics["val_auroc"].update(preds, labels.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.metrics["val_auroc"], on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits, loss, preds, labels = self._common_step(batch)
        # Update all test metrics
        self.metrics["test_auroc"].update(preds, labels.int())
        self.metrics["test_f1"].update(preds, labels.int())
        self.metrics["test_accuracy"].update(preds, labels.int())
        # Log metrics for progress bar
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_auroc", self.metrics["test_auroc"], on_epoch=True, prog_bar=True)
        # Store outputs for detailed report
        self.test_step_outputs.append({"preds": preds.cpu(), "labels": labels.cpu()})

    def on_test_epoch_end(self):
        """Hook to generate the final report at the end of the test phase."""
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        # Binarize predictions for classification report
        preds_binarized = (all_preds > 0.5).int()

        # Generate detailed report using scikit-learn
        report = classification_report(
            all_labels,
            preds_binarized,
            target_names=self.hparams.label_names,
            zero_division=0,
        )

        # Print and save the report
        print("\n" + "=" * 80)
        print(" " * 25 + "DETAILED TEST EVALUATION REPORT")
        print("=" * 80)
        print(report)
        with open("evaluation_report.txt", "w") as f:
            f.write("Evaluation Report\n")
            f.write("=" * 20 + "\n")
            f.write(f"Macro AUROC: {self.metrics['test_auroc'].compute().item():.4f}\n")
            f.write(f"Macro F1-Score: {self.metrics['test_f1'].compute().item():.4f}\n")
            f.write(
                f"Macro Accuracy: {self.metrics['test_accuracy'].compute().item():.4f}\n\n"
            )
            f.write(report)
        print(f"Report saved to evaluation_report.txt")
        print("=" * 80)

        # Important: clear the stored outputs
        self.test_step_outputs.clear()

    def _common_step(self, batch):
        """Helper function to avoid code repetition."""
        image = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(image, input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return logits, loss, preds, labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
