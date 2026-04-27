import torch
import torch.nn as nn
import os
from typing import Optional, List  # CRITICAL for torch.jit.script


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

    # Added explicit type hints for JIT Scripting
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck layer
        x_bottleneck = self.bottleneck(x)

        # TorchScript friendly iteration over ModuleList
        x_convs: List[torch.Tensor] = []
        for conv in self.convs:
            out = conv(x_bottleneck)
            # Use .size(2) instead of .shape[2] for better JIT compatibility
            x_convs.append(out[:, :, :x_bottleneck.size(2)])

        # MaxPooling and 1x1 conv
        x_pool = self.convpool(self.maxpool(x))
        x_convs.append(x_pool)

        # Concatenate outputs along channel dimension
        x_out = torch.cat(x_convs, dim=1)

        # Batch normalization and ReLU
        x_out = self.batchnorm(x_out)
        x_out = self.relu(x_out)

        return x_out


class InceptionTimeModel(nn.Module):
    def __init__(self, num_blocks=3, in_channels=12, num_classes=1000, bottleneck_channels=32, out_channels=32,
                 num_additional_features=0):
        super(InceptionTimeModel, self).__init__()

        num_cov = 4

        self.inception_blocks = nn.Sequential(*[InceptionBlock(in_channels if i == 0 else num_cov * out_channels,
                                                               bottleneck_channels=bottleneck_channels,
                                                               out_channels=out_channels)
                                                for i in range(num_blocks)])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_cov * out_channels + num_additional_features, num_classes)

    # CRITICAL FIX: explicit Optional[torch.Tensor] prevents the 'undefined Tensor' error
    def forward(self, x: torch.Tensor, additional_features: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def serialize_pytorch_inception_time_model(model_params, output_model_filename, device='cpu'):
    net = InceptionTimeModel(
        num_blocks=model_params.get('num_blocks', 3),
        in_channels=model_params.get('in_channels', 12),
        num_classes=model_params.get('num_classes', 1000)
    )

    # load model state dict
    model_state_dict = torch.load(model_params['model_filename_to_use'], map_location=torch.device(device))
    model_state_dict = model_state_dict[
        'model_state_dict'] if 'model_state_dict' in model_state_dict.keys() else model_state_dict

    # create instances of individual model
    net.load_state_dict(model_state_dict)
    net.to(torch.device(device))
    model = net
    model.eval()

    # Now using torch.jit.script successfully!
    traced_script_module = torch.jit.script(model)

    # Prepare extra metadata to embed in the .pt file
    extra_files = {'version': f'{model_params["version"]}',
                   'model_filename_to_use': model_params['model_filename_to_use']}

    # Save the scripted model
    traced_script_module.save(output_model_filename, _extra_files=extra_files)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inception_model_params = {
        'version': '1.0.0',
        'in_channels': 12,
        'num_classes': 1000,
        'num_blocks': 3,
        'model_filename_to_use': 'inception_12x400_1000classes.pt',
        'serialized_model_filename_to_use': './serialized_models/serialized_inception_time_model.pt'
    }

    os.makedirs('./serialized_models_' + device, exist_ok=True)

    output_serialized_model_name = inception_model_params["serialized_model_filename_to_use"].replace(
        "serialized_models", f"serialized_models_{device}")

    if os.path.exists(inception_model_params['model_filename_to_use']):
        serialize_pytorch_inception_time_model(inception_model_params, output_serialized_model_name, device=device)
        print(f"Serialization complete using JIT Script. Saved to: {output_serialized_model_name}")
    else:
        print(f"Error: Could not find the input weights file '{inception_model_params['model_filename_to_use']}'.")