import mne, os

dataset_path = r"C:\Users\nevin\OneDrive\Desktop\EEGMAT_dataset\processed_data"

files = [f for f in os.listdir(dataset_path) if f.endswith('_processed.fif')]
print("Found:", len(files), "files:", files[:5])

raw = mne.io.read_raw_fif(os.path.join(dataset_path, files[0]), preload=True)
print("Channels:", len(raw.ch_names))
print("Sampling rate:", raw.info['sfreq'])
