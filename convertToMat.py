import os
import pyedflib
import numpy as np
from scipy.io import savemat

# Input and output folders
input_folder = "EEGMAT/data"
output_folder = "EEGMAT/converted"

# Make output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over all .edf files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".edf"):
        edf_path = os.path.join(input_folder, filename)
        mat_path = os.path.join(output_folder, filename.replace(".edf", ".mat"))

        print(f"Converting: {filename}")

        try:
            f = pyedflib.EdfReader(edf_path)

            n_signals = f.signals_in_file
            signal_labels = f.getSignalLabels()
            nsamples = f.getNSamples()[0]
            signals = np.zeros((n_signals, nsamples))

            for i in range(n_signals):
                signals[i, :] = f.readSignal(i)

            data_dict = {"signals": signals, "labels": signal_labels}
            savemat(mat_path, data_dict)

            f.close()
            print(f"Saved: {mat_path}")

        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("All EDF files converted successfully!")
