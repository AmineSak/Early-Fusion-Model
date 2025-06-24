# train_lightning.py (Final Version)

import ssl

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from dataloaders.dataloader_cxr import OpenIMultimodalDataset

# Import your updated classes
from models.clf_model import LitEarlyFusionModel

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Configuration ---
config = {
    "full_data_path": "openi_processed_labels.csv",  # The single CSV file
    "train_split": 0.70,  # 70% for training
    "val_split": 0.15,  # 15% for validation
    "test_split": 0.15,  # 15% for testing
    "random_seed": 42,
    "learning_rate": 1e-5,
    "feature_dim": 1024,
    "dropout_p": 0.5,
    "batch_size": 16,
    "max_epochs": 5,
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
}


# --- LightningDataModule with 3-way split ---
class OpenIDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_df, self.val_df, self.test_df = None, None, None

    def setup(self, stage=None):
        full_df = pd.read_csv(self.config["full_data_path"])

        # Split into train+val and test
        train_val_df, self.test_df = train_test_split(
            full_df,
            test_size=self.config["test_split"],
            random_state=self.config["random_seed"],
        )

        # Split train+val into train and val
        val_proportion_of_remainder = self.config["val_split"] / (
            self.config["train_split"] + self.config["val_split"]
        )
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=val_proportion_of_remainder,
            random_state=self.config["random_seed"],
        )

        print(
            f"Data split: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test samples."
        )

    def _create_dataloader(self, df, shuffle=False):
        dataset = OpenIMultimodalDataset(
            data_frame=df, tokenizer=self.tokenizer, image_transform=self.transform
        )
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_df, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_df)

    def test_dataloader(self):
        return self._create_dataloader(self.test_df)


# --- Main Training & Evaluation Execution ---
if __name__ == "__main__":
    # Get label names and number of classes from the data
    full_df = pd.read_csv(config["full_data_path"])
    non_label_cols = ["image_path", "report_text"]
    label_names = [col for col in full_df.columns if col not in non_label_cols]
    num_classes = len(label_names)

    # Initialize DataModule
    data_module = OpenIDataModule(config=config)

    # Initialize Model
    model = LitEarlyFusionModel(
        num_classes=num_classes,
        label_names=label_names,  # Pass the label names for the report
        learning_rate=config["learning_rate"],
        feature_dim=config["feature_dim"],
        dropout_p=config["dropout_p"],
        clinical_bert_name=config["model_name"],
    )

    # --- Setup Callbacks ---
    # This callback saves the best model checkpoint based on validation AUROC
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_auroc:.2f}",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=True,  # Enables logging (e.g., to TensorBoard)
    )

    # --- Run Training ---
    print("\n" + "=" * 80 + "\n" + " " * 30 + "STARTING TRAINING" + "\n" + "=" * 80)
    trainer.fit(model, datamodule=data_module)
    print("\n" + "=" * 80 + "\n" + " " * 31 + "TRAINING COMPLETE" + "\n" + "=" * 80)

    # --- Run Testing ---
    # The .test() method will automatically load the best checkpoint saved during training
    print(
        "\n" + "=" * 80 + "\n" + " " * 28 + "STARTING FINAL TESTING" + "\n" + "=" * 80
    )
    trainer.test(datamodule=data_module, ckpt_path="best")
