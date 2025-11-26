
import tonic
import os
import numpy as np

file_path = "./custom_events/your_gesture.aedat"

print(f"Checking {file_path}...")
if not os.path.exists(file_path):
    print("File not found!")
    exit(1)

try:
    # Try using tonic's file reader if available, or just check file size
    # Tonic might not have a direct 'read_aedat2' exposed easily without a dataset class, 
    # but let's check if we can import something.
    import tonic.io
    events = tonic.io.read_aedat2(file_path)
    print(f"Successfully loaded events. Shape: {events.shape}")
    print(f"Sample events: {events[:5]}")
except Exception as e:
    print(f"Failed to load with tonic: {e}")
    # Fallback: check if it's a binary file we can parse manually or if we need another tool
    print("File size:", os.path.getsize(file_path))
