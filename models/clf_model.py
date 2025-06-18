import torch
from torch import nn
from torchvision import models
from transformers import BertModel


class EarlyFusionMultimodelModel(nn.Module):
    def __init__(self, num_classes, feature_dim, p) -> None:
        super().__init__()
        # Image features exctraction model
        self.image_model = models.resnet50(pretrained=True)

        # Get the dimension of the features before the original fully connected layer
        num_ftrs = self.image_model.fc.in_features

        # Replace the final layer with a new one that outputs our desired feature dimension
        self.image_model.fc = nn.Linear(num_ftrs, feature_dim)

        # Text report features extraction model
        self.report_model = BertModel.from_pretrained("bert-base-uncased")

        # We need another linear layer to project BERT's output (768) to our desired feature_dim (1024)
        self.text_projector = nn.Linear(
            self.report_model.config.hidden_size, feature_dim
        )

        self.clf_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.Relu(),
            nn.Dropout(p),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
        text_outputs = self.report_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        cls_token_output = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projector(cls_token_output)

        fused_features = torch.cat((image_features, text_features))

        logits = self.classifier(fused_features)  # Shape: (batch_size, num_classes)

        return logits
