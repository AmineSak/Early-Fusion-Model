# dataloader_cxr.py (Modified Version)

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# No need to import transformers here, as the tokenizer is passed in


class OpenIMultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for the OpenI multimodal dataset.
    MODIFIED to accept a pandas DataFrame directly instead of a file path.
    """

    def __init__(
        self,
        data_frame,
        tokenizer,
        image_transform=None,
        non_label_cols=["image_path", "report_text"],
    ):
        """
        Args:
            data_frame (pd.DataFrame): The DataFrame containing the data.
            tokenizer: A Hugging Face tokenizer instance.
            image_transform (callable, optional): Optional transform to be applied on an image.
            non_label_cols (list): List of columns that are not labels.
        """
        # --- MODIFICATION ---
        self.data_frame = data_frame
        # --- END MODIFICATION ---

        self.tokenizer = tokenizer
        self.image_transform = image_transform

        self.label_columns = [
            col for col in self.data_frame.columns if col not in non_label_cols
        ]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # We need to use .iloc to access rows by integer index in the passed DataFrame
        row = self.data_frame.iloc[idx]

        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError) as e:
            print(f"ERROR loading image {img_path}: {e}")
            # In a real scenario, you might want to return None and handle it in the collate_fn
            # For now, let's just create a dummy black image to avoid crashing
            return {
                "image": torch.zeros(3, 224, 224),
                "input_ids": torch.zeros(512, dtype=torch.long),
                "attention_mask": torch.zeros(512, dtype=torch.long),
                "labels": torch.zeros(len(self.label_columns), dtype=torch.float32),
            }

        if self.image_transform:
            image = self.image_transform(image)

        report_text = row["report_text"]
        tokenized_text = self.tokenizer(
            report_text,
            padding="max_length",
            truncation=True,
            max_length=512,  # A good standard max length
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)

        labels = row[self.label_columns].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
