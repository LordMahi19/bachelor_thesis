# DVSGesture Dataset Structure

This document outlines the directory structure and data format of the DVSGesture dataset.

## Overview

The dataset is divided into two main parts: training data and testing data, which are stored in the `ibmGestureTrain` and `ibmGestureTest` directories, respectively. The dataset also includes `.tar.gz` archives for both training and testing sets.

## Directory Structure

The data is organized in a hierarchical structure:

```
DVSGesture/
├── ibmGestureTest/
│   ├── user24_fluorescent/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── user24_fluorescent_led/
│   │   └── ...
│   └── ...
├── ibmGestureTrain/
│   ├── user01_fluorescent/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   ├── user01_fluorescent_led/
│   │   └── ...
│   └── ...
├── ibmGestureTest.tar.gz
└── ibmGestureTrain.tar.gz
```

### Levels of the Hierarchy:

1.  **Root Level (`DVSGesture/`)**: Contains the training and testing sets.
2.  **Set Level (`ibmGestureTrain/`, `ibmGestureTest/`)**: Separates the data into training and testing sets.
3.  **User and Lighting Level (`userXX_lighting/`)**: Each directory at this level corresponds to a specific user (`userXX`) and a specific lighting condition (e.g., `fluorescent`, `fluorescent_led`, `led`, `lab`, `natural`).
    *   **Training Set**: Contains data from `user01` to `user23`.
    *   **Testing Set**: Contains data from `user24` to `user29`.
4.  **Gesture Level (`.npy` files)**: Inside each user-lighting directory, there are several `.npy` files, each named with a number (e.g., `0.npy`, `1.npy`, ..., `10.npy`). These numbers represent the gesture classes.

## Data Format

The gesture data is stored in `.npy` files. This format is used for saving single NumPy arrays to disk. Each `.npy` file contains the event data for a single gesture recording.

The `.npy` files contain a structured NumPy array with the following fields for each event:

*   **t**: Timestamp of the event (integer).
*   **x**: x-coordinate of the event (integer).
*   **y**: y-coordinate of the event (integer).
*   **p**: Polarity of the event (0 or 1).

## Summary

To access a specific gesture recording, you would navigate through the following path:

`DVSGesture/<set>/<user_lighting>/<gesture_class>.npy`

For example:

*   `DVSGesture/ibmGestureTrain/user01_fluorescent/0.npy`: Training data for user 1, under fluorescent lighting, for gesture class 0.
*   `DVSGesture/ibmGestureTest/user24_led/10.npy`: Testing data for user 24, under LED lighting, for gesture class 10.
