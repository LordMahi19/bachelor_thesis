import os
import numpy as np
import glob
from pathlib import Path
import h5py
# We might need aedat library if v2e outputs aedat, but v2e often outputs h5 or aedat.
# Let's check what v2e outputs by default or if we can force it.
# The note says "v2e outputs .aedat or .h5".
# If we look at v2e arguments, we can usually specify output format.
# But for now, let's assume we might get .aedat4 (AEDAT 3.1) or .h5.
# Actually, v2e usually outputs .aedat4 by default.
# Reading .aedat4 might require 'aedat' package or 'dv-processing'.
# However, v2e also has an option --output_width and --output_height which we set via --dvs128.
# Let's try to use a simple reader or see if we can export to h5 which is easier to read.
# Wait, v2e has --no_preview to speed up.
# Let's check if we can output h5.
# If not, we will need a way to read aedat.
# For now, I will write a placeholder script that I will refine after seeing the output of the first step.
# But I can write the structure.

def convert_to_npy():
    events_dir = Path("./newdata/generated_events")
    output_dir = Path("./newdata/custom_events_npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We expect folders like ./newdata/generated_events/1/
    # Inside there should be the event file.
    
    subdirs = sorted([d for d in events_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(subdirs)} event folders.")
    
    for subdir in subdirs:
        video_name = subdir.name # e.g. "1"
        
        # Look for .aedat4 or .h5 files
        aedat_files = list(subdir.glob("*.aedat4"))
        h5_files = list(subdir.glob("*.h5"))
        
        input_file = None
        if aedat_files:
            input_file = aedat_files[0]
        elif h5_files:
            input_file = h5_files[0]
            
        if not input_file:
            print(f"No event file found in {subdir}")
            continue
            
        print(f"Processing {input_file}...")
        
        # TODO: Implement actual reading logic based on file type
        # For now, this is a placeholder to be filled after we verify the output format
        # and available libraries.
        
        # Target format: structured array with fields ['t', 'x', 'y', 'p']
        # t: timestamp (us), x, y: coords, p: polarity (0/1)
        
        # Example dummy data creation to show structure
        # events = np.zeros(100, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', '<u1')])
        
        # Save as .npy
        # output_path = output_dir / f"{video_name}.npy"
        # np.save(output_path, events)
        # print(f"Saved to {output_path}")

if __name__ == "__main__":
    convert_to_npy()
