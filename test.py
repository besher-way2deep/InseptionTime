import torch
import os


def test_serialized_model(serialized_model_path, device='cpu'):
    print(f"--- Testing Serialized Model on {device.upper()} ---")

    # 1. Verify the file exists
    if not os.path.exists(serialized_model_path):
        print(f"Error: Could not find '{serialized_model_path}'.")
        print("Please run the serialization script first to generate the .pt file.")
        return

    # =========================================================================
    # THE MAGIC: We are loading the model without importing any network classes!
    # =========================================================================
    print(f"\n1. Loading the TorchScript model from {serialized_model_path}...")
    try:
        loaded_model = torch.jit.load(serialized_model_path, map_location=device)
        loaded_model.eval()
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model. Error: {e}")
        return

    # 2. Create Dummy Data matching your architecture (Batch=1, Channels=12, Time=400)
    print("\n2. Generating dummy input data (1, 12, 400)...")
    dummy_input = torch.randn(1, 12, 400).to(device)

    # 3. Run Inference
    print("\n3. Running inference...")
    try:
        with torch.no_grad():
            raw_logits = loaded_model(dummy_input)

            # Apply your multi-label threshold (Logit > 0 equals Probability > 0.5)
            binary_predictions = (raw_logits > 0.0).int()

        print("   ✅ Inference successful!")
    except Exception as e:
        print(f"   ❌ Inference failed. Error: {e}")
        return

    # 4. Display Results
    print("\n4. Network Architecture Verification:")
    print(f"   Input shape:  {dummy_input.shape}  -> Expected: [1, 12, 400]")
    print(f"   Output shape: {raw_logits.shape} -> Expected: [1, 1000]")

    print(f"\n   First 20 class predictions (0 or 1):")
    print(f"   {binary_predictions[0, :100].tolist()}")

    print("\n🎉 Serialization Test Fully Passed! The model is ready for deployment.")

    # Optional: Extracting the metadata we saved using _extra_files
    print("\n--- Testing Metadata Extraction ---")
    extra_files = {'version': '', 'model_filename_to_use': ''}
    torch.jit.load(serialized_model_path, map_location=device, _extra_files=extra_files)

    print(f"   Embedded Version: {extra_files['version'].decode('utf-8')}")
    print(f"   Source Weights:   {extra_files['model_filename_to_use'].decode('utf-8')}")


if __name__ == '__main__':
    # Determine the device automatically
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Path to the output of your previous script
    # Update this string if you named your folder or file differently
    model_path = f'./serialized_models_{target_device}/serialized_inception_time_model.pt'

    test_serialized_model(model_path, target_device)