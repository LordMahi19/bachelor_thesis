# DVS128 Gesture Dataset Specifications & Injection Notes

## 1. Dataset Specifications
*   **Name:** DVS128 Gesture (IBM Gesture)
*   **Sensor Size:** 128 x 128 pixels
*   **Polarities:** 2 (On/Off)
*   **Time Resolution:** Events are temporal. For training/inference, we accumulate them into frames.
*   **Frame Shape (Processed):** `(Batch, TimeBins, Channels, Height, Width)` -> `(B, 60, 2, 128, 128)`
*   **Classes:** 11 classes (0-10)
*   **Target Class:** **"Air drums" (Index 8)**

## 2. File Structure & Location
*   **Root Directory:** `./newdata/DVSGesture`
*   **Test Set Directory:** `./newdata/DVSGesture/ibmGestureTest`
*   **File Format:** **`.npy` (NumPy Array)**
    *   The dataset is NOT in `.aedat` format as previously thought.
    *   Each gesture instance is a separate `.npy` file.
*   **Directory Hierarchy:**
    ```
    DVSGesture/
    ├── ibmGestureTest/
    │   ├── user24_fluorescent/
    │   │   ├── 0.npy  (Class 0: Hand clapping)
    │   │   ├── ...
    │   │   └── 10.npy (Class 10: Other)
    │   └── ...
    ```
*   **Data Fields (in .npy):** Structured array with fields `['t', 'x', 'y', 'p']`.

## 3. v2e Conversion Parameters
To match the DVS128 sensor, we must use the following settings in `v2e`:
*   **Output Resolution:** 128x128.
*   **DVS Model:** `dvs128`.
*   **Arguments:**
    *   `--dvs128`
    *   `--pos_thres=0.15`
    *   `--neg_thres=0.15`
    *   `--sigma_thres=0.03`
    *   `--cutoff_hz=15`
    *   `--timestamp_resolution=0.003`
*   **Output Format:** `v2e` outputs `.aedat` or `.h5`. We will need a **post-processing step** to convert the `v2e` output to the expected `.npy` format.

## 4. Injection Strategy
We want to add custom "Air drums" samples to the test set.

### Recommended Strategy: Custom Dataset Class
*   Create a `CustomGestureDataset` class that inherits from `tonic.Dataset`.
*   **Loading Logic:**
    1.  Load the original DVSGesture test set (which reads the `.npy` files).
    2.  Load our custom data from a separate directory (e.g., `custom_events/npy/`).
    3.  Our custom data must be converted to the same `.npy` structured array format.
    4.  Combine them into a single dataset for evaluation.

## 5. Action Plan for Data Conversion
1.  **Record RGB Video:** `my_gesture.mp4`
2.  **Convert to Events:** Run `v2e` -> outputs `my_gesture.aedat`.
3.  **Convert to Numpy:** Write a script to read `my_gesture.aedat` (using `tonic` or `aedat` library) and save it as `8.npy` (since Air Drums is class 8).
4.  **Structure:** Place it in a folder structure like `custom_data/user_custom/8.npy` to mimic the original dataset if needed, or just load it directly.
