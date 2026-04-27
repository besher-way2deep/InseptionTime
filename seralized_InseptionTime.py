import torch
import os

# Import the base model and the inference wrapper
from InseptionTime1000Classes import InceptionTimeModel, InceptionTimeInferenceWrapper


def serialize_pytorch_inception_time_model(model_params, output_model_filename, device='cpu'):
    # 1. Initialize the base model
    base_net = InceptionTimeModel(
        num_blocks=model_params.get('num_blocks', 3),
        in_channels=model_params.get('in_channels', 12),
        num_classes=model_params.get('num_classes', 1000)
    )

    # 2. Load the trained weights into the base model
    model_state_dict = torch.load(model_params['model_filename_to_use'], map_location=torch.device(device))
    model_state_dict = model_state_dict[
        'model_state_dict'] if 'model_state_dict' in model_state_dict.keys() else model_state_dict
    base_net.load_state_dict(model_state_dict)
    base_net.to(torch.device(device))
    base_net.eval()

    # 3. Wrap the base model to force 0 and 1 outputs (Int8)
    deployment_model = InceptionTimeInferenceWrapper(base_net)
    deployment_model.eval()

    # 4. Generate example input for tracing
    example = torch.rand(1, model_params.get('in_channels', 12), 400).to(device)

    # 5. Serialize the WRAPPED model using JIT Trace
    traced_script_module = torch.jit.trace(deployment_model, example)

    # 6. Prepare extra metadata and save
    extra_files = {'version': f'{model_params["version"]}',
                   'model_filename_to_use': model_params['model_filename_to_use']}
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
        print(
            f"Serialization complete. The deployment model (Int8 0/1 outputs) is saved to: {output_serialized_model_name}")
    else:
        print(
            f"Error: Could not find the input weights file '{inception_model_params['model_filename_to_use']}'. Please run the training script first.")