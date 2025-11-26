# Title
link to video presentaion:

https://lut-my.sharepoint.com/:v:/g/personal/mahi_talukder_student_lut_fi/EdtPBJR2bv9PjGYuD2swwo4BSeHDd1_Lzf6o5iqoUABx6w?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=ynb6wC

Improving Robustness of Event-Based Models
# Thesis Project: Investigating Event-Based Model Robustness

## Research Goal

This project investigates the robustness of a pre-trained 3D-CNN event-based action recognition model. The core hypothesis is that a model trained on clean, lab-recorded data (DVS128 Gesture) will see a performance drop when evaluated against custom-recorded, real-world data containing variations in angle, lighting, and background.

The research aims to answer the following question: **To what extent can this performance drop be mitigated by augmenting the training dataset with new, custom-recorded real-world data?**

This research direction is a pivot from a previous, broader goal of comparing RGB and event-based cameras, and was chosen to create a more focused and less complicated thesis project.

## Project Plan

The project is broken down into six distinct phases to systematically test the hypothesis.

### Phase 1: Baseline Model Evaluation (Completed)

1.  **Load Pre-trained Model:**
    *   Load the `Gesture3DCNN` model which has been previously trained on the DVS128 Gesture dataset.
    *   The model definition is in `model_def.py`.

2.  **Evaluate on Original Test Set:**
    *   Run inference on the original DVS128 Gesture test set.
    *   The results of this are logged in `baseline_test_log.md`, establishing the **Baseline Performance** with ~84% accuracy.

### Phase 2: Custom Data Collection and Conversion (In Progress)

1.  **Identify Target Gestures:**
    *   List the specific gestures from the DVS128 dataset to be re-recorded.

2.  **Record Custom Videos:**
    *   Using a standard RGB camera, record multiple instances of each gesture, introducing variations in:
        *   **Angles:** Different viewpoints (front, side, above, below).
        *   **Distances:** Close, medium, and far.
        *   **Lighting:** Bright, dim, and uneven lighting.
        *   **Backgrounds:** Cluttered and plain.
        *   **Speed:** Fast, normal, and slow execution.

3.  **Convert Videos to Event Data using `v2e`:**
    *   Use the `v2e` tool to convert the recorded RGB videos into event streams (`.aedat4` format).
    *   A detailed guide for this process is in `execution.md`.

### Phase 3: Robustness Test 1 - Performance Degradation

1.  **Create Augmented Test Set:**
    *   Combine the original DVS128 Gesture test set with the newly generated custom event data.

2.  **Evaluate Baseline Model on Augmented Test Set:**
    *   Load the *original* pre-trained `Gesture3DCNN` model.
    *   Run inference on the new "Augmented Test Set."
    *   Collect metrics and compare them with the "Baseline Performance" to quantify the expected performance drop.

### Phase 4: Data Augmentation and Re-training

1.  **Create Augmented Training Set:**
    *   Combine the original DVS128 Gesture training set with the newly generated custom event data.

2.  **Re-train the Model:**
    *   Train a new `Gesture3DCNN` model on the "Augmented Training Set."
    *   Save the best-performing model checkpoint.

### Phase 5: Robustness Test 2 - Performance Improvement

1.  **Evaluate Re-trained Model on Augmented Test Set:**
    *   Load the newly re-trained model from Phase 4.
    *   Run inference on the *same* "Augmented Test Set" from Phase 3.
    *   Collect metrics and compare the results to quantify any performance improvement.

### Phase 6: Analysis and Conclusion

1.  **Compare Results:**
    *   Analyze the three sets of performance metrics:
        1.  Baseline Performance (Original model, original test set)
        2.  Degraded Performance (Original model, augmented test set)
        3.  Improved Performance (Re-trained model, augmented test set)

2.  **Draw Conclusions:**
    *   Discuss whether augmenting the training data with real-world samples improved the model's generalization.
    *   Address the initial hypothesis and write up the findings for the thesis.
