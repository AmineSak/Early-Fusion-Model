import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenIMultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for the OpenI multimodal dataset.

    This class loads an image from a path, tokenizes a text report,
    and retrieves the corresponding multi-label vector.
    """

    def __init__(
        self,
        csv_file,
        tokenizer,
        image_transform=None,
        non_label_cols=["image_path", "report_text"],
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer: A Hugging Face tokenizer instance.
            image_transform (callable, optional): Optional transform to be applied on an image.
            non_label_cols (list): List of columns that are not labels.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_transform = image_transform

        # Identify label columns
        self.label_columns = [
            col for col in self.data_frame.columns if col not in non_label_cols
        ]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Fetches the a single sample from the dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A dictionary containing:
            - 'image': The transformed image tensor.
            - 'input_ids': The tokenized text input IDs.
            - 'attention_mask': The text attention mask.
            - 'labels': The multi-label target tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load Image
        img_path = self.data_frame.loc[idx, "image_path"]
        # Use a try-except block in case an image file is missing or corrupt
        try:
            image = Image.open(img_path).convert(
                "RGB"
            )  # Convert to RGB to ensure 3 channels
        except (FileNotFoundError, IOError):
            # If an image is bad, you might want to return a dummy tensor
            # or skip this sample. For simplicity, we'll raise an error.
            print(f"Warning: Could not load image {img_path}. Skipping.")
            # A better approach for production is to handle this gracefully
            # by returning a dummy item or having a list of valid indices.
            return None

        if self.image_transform:
            image = self.image_transform(image)

        # 2. Process Text
        report_text = self.data_frame.loc[idx, "report_text"]
        # The tokenizer returns a dictionary with 'input_ids' and 'attention_mask'
        tokenized_text = self.tokenizer(
            report_text,
            padding="max_length",  # Pad to a fixed length
            truncation=True,  # Truncate if longer than max_length
            max_length=500,  # A reasonable max length for reports
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Squeeze to remove the batch dimension added by the tokenizer
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)

        # 3. Get Labels
        labels = self.data_frame.loc[idx, self.label_columns].values
        labels = torch.tensor(
            labels, dtype=torch.float32
        )  # Must be float for BCEWithLogitsLoss

        # 4. Return as a dictionary
        sample = {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return sample
