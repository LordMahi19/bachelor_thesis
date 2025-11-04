# export_and_infer.py
import os, json
import torch
import numpy as np
import tonic
import tonic.transforms as transforms

from model_def import Gesture3DCNN

# ---------- Paths & constants ----------
DATA_DIR     = "./newdata"                       # where DVSGesture lives
RAW_BEST_CKPT= "./best_dvsgesture_3dcnn.pth"     # produced by your training code
SAVE_DIR     = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)

COMBINED_CKPT= os.path.join(SAVE_DIR, "dvsgesture_3dcnn_best.pth")
CONFIG_JSON  = os.path.join(SAVE_DIR, "dvsgesture_config.json")
TS_PATH      = os.path.join(SAVE_DIR, "dvsgesture_3dcnn_scripted.pt")
ONNX_PATH    = os.path.join(SAVE_DIR, "dvsgesture_3dcnn.onnx")

# Same settings as training
NUM_CLASSES  = 11
N_FRAMES     = 60
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZATION= "per-sample-max"  # you used per-sample max normalization

# ---------- Helper: ensure sensor_size is a 3-tuple (H, W, C) ----------
def ensure_3d_sensor_size(sensor_size):
    """Accepts (H, W) or (H, W, C); returns (H, W, C) with C=2 if missing."""
    if len(sensor_size) == 2:
        return (sensor_size[0], sensor_size[1], 2)
    return tuple(sensor_size)

# ---------- Read sensor size from the dataset (authoritative) ----------
try:
    SENSOR_SIZE_3D = ensure_3d_sensor_size(tuple(tonic.datasets.DVSGesture.sensor_size))
except Exception:
    # Fallback if dataset attr not reachable
    SENSOR_SIZE_3D = (128, 128, 2)

H, W, C = SENSOR_SIZE_3D

# ---------- Class names ----------
try:
    tmp_ds = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=True)
    CLASS_NAMES = getattr(tmp_ds, "classes", [str(i) for i in range(NUM_CLASSES)])
except Exception:
    CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# ---------- 1) Load raw best checkpoint (from training) ----------
if not os.path.exists(RAW_BEST_CKPT):
    raise FileNotFoundError(
        f"Could not find {RAW_BEST_CKPT}. Run your training script first so it saves the best checkpoint."
    )

raw = torch.load(RAW_BEST_CKPT, map_location="cpu")  # {'state_dict':..., 'acc':...}
state_dict = raw["state_dict"]

model = Gesture3DCNN(num_classes=NUM_CLASSES)
model.load_state_dict(state_dict)
model.eval()

# ---------- 2) Save combined checkpoint WITH metadata ----------
combined = {
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "num_classes": NUM_CLASSES,
    "n_frames": N_FRAMES,
    "sensor_size": SENSOR_SIZE_3D,    # <-- 3-tuple saved
    "normalization": NORMALIZATION,
    "class_names": CLASS_NAMES,
    "best_acc": float(raw.get("acc", -1.0)),
}
torch.save(combined, COMBINED_CKPT)
print(f"[OK] Saved combined checkpoint with metadata -> {COMBINED_CKPT}")

# ---------- 3) Save config JSON (human-friendly) ----------
with open(CONFIG_JSON, "w") as f:
    json.dump({
        "num_classes": NUM_CLASSES,
        "n_frames": N_FRAMES,
        "sensor_size": SENSOR_SIZE_3D,  # <-- 3-tuple saved
        "normalization": NORMALIZATION,
        "class_names": CLASS_NAMES,
    }, f, indent=2)
print(f"[OK] Saved config JSON -> {CONFIG_JSON}")

# ---------- 4) Export TorchScript ----------
scripted = torch.jit.script(model.cpu().eval())
scripted.save(TS_PATH)
print(f"[OK] Saved TorchScript model -> {TS_PATH}")

# ---------- 5) Export ONNX ----------
dummy = torch.randn(1, 2, N_FRAMES, H, W, dtype=torch.float32)  # (B,C,T,H,W)

# You can pass dynamo=True on newer PyTorch to use the new exporter:
# torch.onnx.export(..., dynamo=True)
torch.onnx.export(
    model.cpu().eval(),
    dummy,
    ONNX_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    # dynamo=True,  # uncomment if your torch supports the new exporter
)
print(f"[OK] Saved ONNX model -> {ONNX_PATH}")

# ---------- 6) Preprocessing & inference (same as training) ----------
frame_transform = transforms.ToFrame(
    sensor_size=SENSOR_SIZE_3D,  # 3-tuple here
    n_time_bins=N_FRAMES
)

def preprocess_event_stream(event_stream):
    """
    event_stream: NumPy structured array (x, y, p, t).
    Returns: (1, 2, T, H, W) float32 normalized to [0,1].
    """
    frames = frame_transform(event_stream)                  # (T, 2, H, W)
    x = torch.tensor(frames).unsqueeze(0)                   # (1, T, 2, H, W)
    x = x.permute(0, 2, 1, 3, 4).float()                    # (1, 2, T, H, W)
    x = x / (x.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)    # per-sample max normalization
    return x

@torch.no_grad()
def predict_with_pytorch(event_stream):
    """Inference using the PyTorch model class (eager)."""
    ckpt = torch.load(COMBINED_CKPT, map_location="cpu")
    num_classes = ckpt["num_classes"]
    state = ckpt["state_dict"]
    class_names = ckpt.get("class_names", [str(i) for i in range(num_classes)])

    mdl = Gesture3DCNN(num_classes=num_classes).to(DEVICE).eval()
    mdl.load_state_dict(state)

    x = preprocess_event_stream(event_stream).to(DEVICE)
    logits = mdl(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(probs.argmax())
    pred_name = class_names[pred_id] if isinstance(class_names, list) and len(class_names)==num_classes else str(pred_id)
    return pred_id, pred_name, probs

@torch.no_grad()
def predict_with_torchscript(event_stream):
    """Inference using the TorchScript model (no Python class needed)."""
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)

    # Backward-compat for old 2-tuple configs:
    sensor_size = ensure_3d_sensor_size(tuple(cfg["sensor_size"]))
    n_frames    = int(cfg["n_frames"])
    class_names = cfg.get("class_names", [str(i) for i in range(cfg["num_classes"])])

    ts_model = torch.jit.load(TS_PATH, map_location=DEVICE).eval()

    tf = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_frames)
    frames = tf(event_stream)  # (T, 2, H, W)
    x = torch.tensor(frames).unsqueeze(0).permute(0, 2, 1, 3, 4).float()
    x = x / (x.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)

    x = x.to(DEVICE)
    logits = ts_model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(probs.argmax())
    pred_name = class_names[pred_id] if isinstance(class_names, list) and len(class_names)==len(class_names) else str(pred_id)
    return pred_id, pred_name, probs

# ---------- 7) Smoke test ----------
if __name__ == "__main__":
    test_ds = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=False)  # raw events (no transform)
    ev, lab = test_ds[0]

    pred_id, pred_name, probs = predict_with_pytorch(ev)
    print(f"[PyTorch]     True: {lab} | Pred: {pred_id} ({pred_name}) | top1: {probs.max():.3f}")

    pred_id2, pred_name2, probs2 = predict_with_torchscript(ev)
    print(f"[TorchScript] True: {lab} | Pred: {pred_id2} ({pred_name2}) | top1: {probs2.max():.3f}")