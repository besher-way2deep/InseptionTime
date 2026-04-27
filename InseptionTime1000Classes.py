import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Retained your original imports, commented out only to allow standalone execution for the mockup
# from InceptionTime import utils
# from InceptionTime import DataLoaderUtils
# from Lilyana import data_loader
from tqdm import tqdm
import matplotlib

matplotlib.use("TkAgg")


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=32, kernel_sizes=[10, 20, 40], out_channels=32):
        super(InceptionBlock, self).__init__()

        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.batchnorm = nn.BatchNorm1d((len(kernel_sizes) + 1) * out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck layer
        x_bottleneck = self.bottleneck(x)

        # Apply different kernel convolutions
        # Modified slicing to dynamically fit sequence length instead of hardcoded 2500
        x_convs = [conv(x_bottleneck)[:, :, :x_bottleneck.shape[2]] for conv in self.convs]

        # MaxPooling and 1x1 conv
        x_pool = self.convpool(self.maxpool(x))

        # Concatenate outputs along channel dimension
        x_out = torch.cat(x_convs + [x_pool], dim=1)

        # Batch normalization and ReLU
        x_out = self.batchnorm(x_out)
        x_out = self.relu(x_out)

        return x_out


class InceptionTimePlus(nn.Module):
    def __init__(self, num_blocks=3, in_channels=12, num_classes=1000, bottleneck_channels=32, out_channels=32,
                 num_additional_features=0):
        super(InceptionTimePlus, self).__init__()

        num_cov = 4

        self.inception_blocks = nn.Sequential(*[InceptionBlock(in_channels if i == 0 else num_cov * out_channels,
                                                               bottleneck_channels=bottleneck_channels,
                                                               out_channels=out_channels)
                                                for i in range(num_blocks)])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_cov * out_channels + num_additional_features, num_classes)  # Adjusted input size

    def forward(self, x, additional_features=None):
        # Pass through the Inception Blocks
        x = self.inception_blocks(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        # If additional features are provided, concatenate them
        if additional_features is not None:
            x = torch.cat((x, additional_features), dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x


def evaluate_model(model, dataloader, device, criterion=nn.BCEWithLogitsLoss()):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_samples, batch_labels in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device).float()

            outputs = model(batch_samples)  # [batch, num_classes]
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_samples.size(0)

            # For multi-label (0 or 1), threshold outputs at 0 (equivalent to sigmoid(x) > 0.5)
            predicted = (outputs > 0.0).float()

            total += batch_labels.numel()  # Total number of elements across all classes and batch
            correct += (predicted == batch_labels).sum().item()

            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=3):
    best_val_loss = float('inf')
    total_train_loss, total_val_loss = [], []
    total_train_acc, total_val_acc = [], []
    counter = 0  # this is a counter for the early stopping algorithm

    # L1 and L2 regularization parameters
    lambda_l1 = 1e-5
    lambda_l2 = 1e-4
    lambda_bias_l2 = 1e-4

    print_model_flag = True

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)

            if print_model_flag:
                print_model_flag = False

            loss = criterion(outputs, labels)

            # Regularization terms
            l1_regularization = 0
            l2_regularization = 0
            l2_bias_regularization = 0

            for module in model.children():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    l1_regularization += torch.norm(module.weight, 1)
                    l2_regularization += torch.norm(module.weight, 2) ** 2
                    if module.bias is not None:
                        l2_bias_regularization += torch.norm(module.bias, 2) ** 2

            # Add regularization terms to the loss
            loss += lambda_l1 * l1_regularization + lambda_l2 * l2_regularization + lambda_bias_l2 * l2_bias_regularization

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Multi-label accuracy calculation
            predicted = (outputs > 0.0).float()
            total_train += labels.numel()
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        total_train_loss.append(train_loss)
        total_train_acc.append(train_acc)

        total_val_loss.append(val_loss)
        total_val_acc.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_best_model_1000_classes_test.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return total_train_loss, total_train_acc, total_val_loss, total_val_acc


def build_confusion_matrix(y_test, y_pred):
    # Note: Traditional confusion matrices are difficult to visualize for 1000 classes.
    # This is retained per constraints, but kept functionally identical.
    # For a multi-label output of shape (N, 1000), flattening or calculating per-class metrics is usually required.
    cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
    print(cm)

    accuracy = np.trace(cm) / np.sum(cm)
    print("Accuracy:", accuracy)
    return


def plot_metric(train_values, val_values, metric_name="Metric", image_name="Metric.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_values, label='Train', marker='o')
    plt.plot(val_values, label='Validation', marker='o')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()


########################################################################################


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    # Network architecture specifications
    batch_size = 64
    num_channels = 12  # ECG leads + manual features
    sequence_length = 400  # Time steps required (12 x 400 input)
    num_classes = 1000  # 1000 independent classes (Multi-label)
    additional_features = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =========================================================================
    # Mockup Data Generation to bypass external loaders and test functionality
    # =========================================================================
    print("Generating Mockup Data for 12x400 input -> 1000 multi-label outputs...")

    # Generate 128 random samples of shape (12, 400)
    mock_X_train = torch.randn(128, num_channels, sequence_length)
    # Generate 128 random multi-label targets of shape (1000,) containing 0 or 1
    mock_y_train = torch.randint(0, 2, (128, num_classes)).float()

    mock_X_val = torch.randn(64, num_channels, sequence_length)
    mock_y_val = torch.randint(0, 2, (64, num_classes)).float()

    train_dataset = TensorDataset(mock_X_train, mock_y_train)
    val_dataset = TensorDataset(mock_X_val, mock_y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = InceptionTimePlus(num_blocks=3, in_channels=num_channels, num_classes=num_classes,
                              num_additional_features=additional_features)
    model = model.to(device)

    # Verify model forward pass with the correct dimensions
    print("Running a forward pass verification...")
    dummy_input = torch.randn(1, num_channels, sequence_length).to(device)
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {dummy_output.shape}")

    # Save the mockup model file
    mockup_filename = "inception_12x400_1000classes.pt"
    torch.save(model.state_dict(), mockup_filename)
    print(f"Mockup state dictionary successfully saved to: {mockup_filename}")

    # =========================================================================
    # Training Loop (Optional - uncomment below to train the mock setup)
    # =========================================================================

    # criterion = nn.BCEWithLogitsLoss() # Changed for Multi-label classification (0 or 1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
    #     model, train_loader, val_loader, criterion, optimizer, device, num_epochs=2, patience=50
    # )

    # plot_metric(total_train_acc, total_val_acc, "Accuracy")
    # plot_metric(total_train_loss, total_val_loss, "Loss")