import torch
import os
import matplotlib.pyplot as plt
import numpy as np


# Optional: If you have display issues on Linux/Mac, uncomment the line below
# import matplotlib
# matplotlib.use("TkAgg")

def test_and_visualize_model(serialized_model_path, device='cpu'):
    print(f"--- Visualizing Serialized Model on {device.upper()} ---")

    if not os.path.exists(serialized_model_path):
        print(f"Error: Could not find '{serialized_model_path}'.")
        return

    # 1. Load the model
    print(f"1. Loading the TorchScript model...")
    loaded_model = torch.jit.load(serialized_model_path, map_location=device)
    loaded_model.eval()

    # 2. Generate Dummy 12-Lead Data
    print("2. Generating simulated 12-lead ECG data...")
    # Adding a sine wave to the random noise so the plot looks like a continuous signal
    time_steps = np.linspace(0, 4 * np.pi, 400)
    dummy_numpy = np.random.randn(12, 400) * 0.1 + np.sin(time_steps)
    dummy_input = torch.tensor(dummy_numpy, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Run Inference
    print("3. Running inference...")
    with torch.no_grad():
        binary_predictions = loaded_model(dummy_input)
        print(binary_predictions)

    # Extract the specific class IDs that were predicted as '1' (Positive)
    # binary_predictions[0] accesses the first item in the batch
    predicted_classes = torch.where(binary_predictions[0] == 1)[0].tolist()

    print(f"\n--- Results ---")
    print(f"Total Positive Triggers: {len(predicted_classes)}")
    print(f"Detected Class IDs: {predicted_classes}")

    # 4. Generate Visualization
    print("\n4. Generating Visualization Plot...")

    # Create a 12-row plot for the 12 leads
    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(10, 14), sharex=True)
    fig.suptitle('InceptionTime Model: 12-Lead Input & Predictions', fontsize=16, fontweight='bold', y=0.98)

    # Plot each lead
    for i in range(12):
        axes[i].plot(dummy_input[0, i, :].cpu().numpy(), color='#1f77b4', linewidth=1.2)
        axes[i].set_ylabel(f'Lead {i + 1}', rotation=0, labelpad=30, va='center', fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time Steps (400)', fontweight='bold', fontsize=12)

    # Add a highlighted text box at the top with the prediction results
    result_text = "Model Outputs (Triggered Classes):\n"
    if not predicted_classes:
        result_text += "None (All classes output 0)"
    else:
        # Wrap text if the model triggers too many classes to fit on one line
        import textwrap
        class_str = ", ".join(map(str, predicted_classes))
        result_text += textwrap.fill(class_str, width=80)

    plt.figtext(0.5, 0.94, result_text, ha="center", fontsize=11,
                bbox={"facecolor": "#d4edda", "edgecolor": "#c3e6cb", "alpha": 0.8, "pad": 8,
                      "boxstyle": "round,pad=0.5"})

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave room for the title and text box

    # Save and Show
    output_image = "visualized_inference_test.png"
    plt.savefig(output_image, dpi=150)
    print(f"Visualization successfully saved to '{output_image}'")
    plt.show()


if __name__ == '__main__':
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Update this path if necessary to point to your final serialized wrapper
    model_path = f'./serialized_models_{target_device}/serialized_inception_time_model.pt'

    test_and_visualize_model(model_path, target_device)