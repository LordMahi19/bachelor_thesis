to recreate my conda environment:

```
conda env create -f environment.yml

```

## Running Inference

This project includes exported models and scripts to run inference on event data. The models are located in the `/models` directory in PyTorch, TorchScript, and ONNX formats.

### Prerequisites

1.  **Conda Environment**: Make sure you have created and activated the conda environment using the `environment.yml` file.
2.  **Models**: The models must be present in the `./models/` directory. If not, you can generate them by running `export_and_infer.py`.
3.  **Event Data**: You need event data in the form of a NumPy structured array with `(x, y, p, t)` fields. The `export_and_infer.py` script demonstrates loading data from the `tonic.datasets.DVSGesture` dataset. You can adapt the inference script to load your own data from a `.npy` file. The `v2e` directory contains tools to convert video into events, which can be a source for new data.

### Inference with a Python Script

You can run inference using the exported TorchScript model (`dvsgesture_3dcnn_scripted.pt`), which is generally the easiest method as it doesn't require the original model class definition.

Here is an example script to run inference on a `.npy` file containing event data. You can save this as `run_inference.py`:

```python
import torch
import tonic.transforms as transforms
import numpy as np
import json
import argparse

# --- Constants ---
MODEL_DIR = "./models"
TS_PATH = f"{MODEL_DIR}/dvsgesture_3dcnn_scripted.pt"
CONFIG_PATH = f"{MODEL_DIR}/dvsgesture_config.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper function to ensure sensor_size is a 3-tuple ---
def ensure_3d_sensor_size(sensor_size):
    """Accepts (H, W) or (H, W, C); returns (H, W, C) with C=2 if missing."""
    if len(sensor_size) == 2:
        return (sensor_size[0], sensor_size[1], 2)
    return tuple(sensor_size)

# --- Main Inference Function ---
@torch.no_grad()
def predict_from_file(event_file_path):
    """
    Loads an event stream from a .npy file, preprocesses it, and runs inference.
    """
    # 1. Load config and model
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    sensor_size = ensure_3d_sensor_size(tuple(cfg["sensor_size"]))
    n_frames = int(cfg["n_frames"])
    class_names = cfg.get("class_names", [str(i) for i in range(cfg["num_classes"])])

    ts_model = torch.jit.load(TS_PATH, map_location=DEVICE).eval()

    # 2. Load event data
    try:
        event_stream = np.load(event_file_path)
        # Basic validation of the structured array
        if not all(field in event_stream.dtype.names for field in ['x', 'y', 't', 'p']):
             raise ValueError("Input .npy file must be a structured array with fields 'x', 'y', 't', 'p'.")
    except Exception as e:
        print(f"Error loading or validating event file: {e}")
        return

    # 3. Preprocess data (same as in export_and_infer.py)
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_frames)
    frames = frame_transform(event_stream)  # (T, 2, H, W)
    x = torch.tensor(frames).unsqueeze(0).permute(0, 2, 1, 3, 4).float() # (1, 2, T, H, W)

    # Per-sample max normalization
    x = x / (x.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
    x = x.to(DEVICE)

    # 4. Run inference
    logits = ts_model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(probs.argmax())
    pred_name = class_names[pred_id]

    print(f"File: {event_file_path}")
    print(f"Predicted Gesture: {pred_id} -> {pred_name}")
    print(f"Confidence: {probs.max():.4f}")
    print("-" * 20)
    # print("Full probabilities:", probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gesture inference on an event file.")
    parser.add_argument("event_file", help="Path to the .npy file containing the event stream.")
    args = parser.parse_args()

    predict_from_file(args.event_file)

```

### How to Use the Script

1.  **Save the script**: Save the code above as `run_inference.py` in the root of the project.
2.  **Get sample data**: To test the script, you can first generate a sample event file from the test dataset. Add the following snippet to the `if __name__ == "__main__":` block in `export_and_infer.py` and run it:

    ```python
    # Inside export_and_infer.py's main block
    import numpy as np
    test_ds = tonic.datasets.DVSGesture(save_to="./newdata", train=False)
    events, label = test_ds[1] # get the second sample (Right hand wave)
    np.save("sample_event_data.npy", events)
    print(f"Saved sample event data with label {label} to sample_event_data.npy")
    ```
    This will create a `sample_event_data.npy` file in your project root.

3.  **Run inference**: Now, run the inference script from your terminal:

    ```bash
    python run_inference.py sample_event_data.npy
    ```

4.  **Expected Output**:

    ```
    File: sample_event_data.npy
    Predicted Gesture: 1 -> Right hand wave
    Confidence: 0.999...
    --------------------
    ```
