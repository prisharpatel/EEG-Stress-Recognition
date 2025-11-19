import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.stats import kurtosis, skew
from scipy.signal import welch, stft
import warnings
warnings.filterwarnings('ignore')


class EEGFeatureExtractor:
    def __init__(self):
        self.features_df = pd.DataFrame()
    
    def find_processed_files(self):
        """Find all processed FIF files in various possible locations"""
        dataset_path = r"C:\Users\nevin\OneDrive\Desktop\EEGMAT_dataset\eeg-during-mental-arithmetic-tasks-1.0.0"
        
        possible_locations = [
            os.path.join(dataset_path, 'processed_data'),
            dataset_path,
            'processed_data',
            os.path.join(os.path.dirname(dataset_path), 'processed_data')
        ]
        
        processed_files = []
        
        for location in possible_locations:
            if os.path.exists(location):
                print(f"Searching in: {location}")
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if file.endswith('_processed.fif'):
                            full_path = os.path.join(root, file)
                            processed_files.append(full_path)
                            print(f"  Found: {file}")
        
        return processed_files
    
    def load_processed_data(self, file_path):
        """Load preprocessed EEG data"""
        try:
            print(f"Loading: {os.path.basename(file_path)}")
            raw = mne.io.read_raw_fif(file_path, preload=True)
            print(f"   Success - {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s")
            return raw
        except Exception as e:
            print(f"   Error loading: {e}")
            return None
    
    def extract_basic_features(self, data, channel_name, sfreq):
        """Extract time-domain, frequency-domain, and time–frequency features"""
        features = {}
        
        # ------------------------------
        # 🧠 TIME-DOMAIN FEATURES
        # ------------------------------
        features[f'{channel_name}_mean'] = np.mean(data)
        features[f'{channel_name}_std'] = np.std(data)
        features[f'{channel_name}_variance'] = np.var(data)
        features[f'{channel_name}_skewness'] = skew(data)
        features[f'{channel_name}_kurtosis'] = kurtosis(data)
        
        # Hjorth parameters
        diff1 = np.diff(data)
        activity = np.var(data)
        mobility = np.std(diff1) / np.std(data)
        complexity = np.std(np.diff(diff1)) / np.std(diff1)
        features[f'{channel_name}_hjorth_activity'] = activity
        features[f'{channel_name}_hjorth_mobility'] = mobility
        features[f'{channel_name}_hjorth_complexity'] = complexity

        # Entropy (time-domain)
        prob, _ = np.histogram(data, bins=50, density=True)
        prob = prob[prob > 0]
        features[f'{channel_name}_entropy'] = -np.sum(prob * np.log2(prob))
        
        # ------------------------------
        # 🎚️ FREQUENCY-DOMAIN FEATURES
        # ------------------------------
        freqs, psd = welch(data, fs=sfreq, nperseg=1024)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        total_power = np.trapz(psd, freqs)
        
        for band, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                relative_power = band_power / total_power if total_power > 0 else 0
                
                features[f'{channel_name}_{band}_power'] = band_power
                features[f'{channel_name}_{band}_relative'] = relative_power
        
        # ------------------------------
        # ⏱️ TIME–FREQUENCY FEATURES (STFT)
        # ------------------------------
        f, t, Zxx = stft(data, fs=sfreq, nperseg=256)
        stft_power = np.abs(Zxx) ** 2
        features[f'{channel_name}_stft_energy_mean'] = np.mean(stft_power)
        features[f'{channel_name}_stft_energy_std'] = np.std(stft_power)
        
        return features
    
    def extract_features_from_file(self, file_path):
        """Extract features from a single processed file"""
        file_name = os.path.basename(file_path)
        
        # Determine task type
        if '_1_' in file_name:
            task_type = "REST"
        elif '_2_' in file_name:
            task_type = "MATH"
        else:
            task_type = "UNKNOWN"
        
        print(f"\nProcessing: {file_name} ({task_type})")
        print("-" * 40)
        
        raw_data = self.load_processed_data(file_path)
        if raw_data is None:
            return None
        
        eeg_channel_names = [ch for ch in raw_data.ch_names if 'EEG' in ch]
        if not eeg_channel_names:
            print("  No EEG channels found!")
            return None
        
        eeg_data = raw_data.copy().pick(eeg_channel_names)
        data_matrix = eeg_data.get_data()
        sfreq = eeg_data.info['sfreq']
        
        features = {
            'file_name': file_name,
            'task_type': task_type,
            'subject_id': file_name.split('_')[0],
            'duration': raw_data.times[-1],
            'num_channels': len(eeg_channel_names),
            'sampling_rate': sfreq
        }
        
        for i, channel_name in enumerate(eeg_channel_names):
            channel_data = data_matrix[i]
            channel_features = self.extract_basic_features(channel_data, channel_name, sfreq)
            features.update(channel_features)
        
        print(f"  Extracted {len(features)} features")
        return features
    
    def run_feature_extraction(self, max_files=5):
        """Main function to run feature extraction"""
        print("STARTING FEATURE EXTRACTION")
        print("=" * 60)
        
        processed_files = self.find_processed_files()
        
        if not processed_files:
            print(" No processed files found!")
            print("\nPlease run preprocessing first using:")
            print("python run_preprocessing.py")
            return None
        
        print(f"\nFound {len(processed_files)} processed files")
        
        all_features = []
        for i, file_path in enumerate(processed_files if max_files is None else processed_files[:max_files]):
            features = self.extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
        
        if not all_features:
            print(" No features extracted from any files!")
            return None
        
        self.features_df = pd.DataFrame(all_features)
        
        output_file = 'eeg_features.csv'
        self.features_df.to_csv(output_file, index=False)
        
        print(f"\n FEATURE EXTRACTION COMPLETED!")
        print(f"   Files processed: {len(self.features_df)}")
        print(f"   Features per file: {len(self.features_df.columns)}")
        print(f"   Output file: {output_file}")
        
        print(f"\nTask Distribution:")
        print(self.features_df['task_type'].value_counts())
        
        return self.features_df


def main():
    extractor = EEGFeatureExtractor()
    features_df = extractor.run_feature_extraction(max_files=None)


if __name__ == "__main__":
    main()
