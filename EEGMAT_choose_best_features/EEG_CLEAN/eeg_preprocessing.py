import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.raw_data = None
        self.filtered_data = None
        self.cleaned_data = None
        
    def load_edf_file(self, file_path):
        """Load EDF file using MNE"""
        try:
            print(f"Loading {os.path.basename(file_path)}")
            self.raw_data = mne.io.read_raw_edf(file_path, preload=True)
            print(f"Successfully loaded: {len(self.raw_data.ch_names)} channels, "
                  f"{self.raw_data.n_times} samples, "
                  f"SF: {self.raw_data.info['sfreq']} Hz")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def basic_info(self):
        """Display basic information about the EEG data"""
        if self.raw_data is None:
            print("No data loaded")
            return
            
        print("\n" + "="*50)
        print("EEG DATA INFORMATION")
        print("="*50)
        print(f"Channels: {len(self.raw_data.ch_names)}")
        print(f"Sampling frequency: {self.raw_data.info['sfreq']} Hz")
        print(f"Duration: {self.raw_data.times[-1]:.2f} seconds")
        print(f"Data shape: {self.raw_data.get_data().shape}")
        print(f"Channel types: {set(self.raw_data.get_channel_types())}")
        
    def apply_filtering(self, l_freq=1.0, h_freq=40.0, notch_freq=50.0):
        """Apply bandpass and notch filtering"""
        if self.raw_data is None:
            print("No data loaded")
            return
            
        print("\nAPPLYING FILTERS")
        print("="*50)
        
        # Create a copy for filtering
        self.filtered_data = self.raw_data.copy()
        
        # Apply bandpass filter (1-40 Hz) - typical EEG range
        print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
        self.filtered_data.filter(l_freq=l_freq, h_freq=h_freq, 
                                 method='iir', verbose=False)
        
        # Apply notch filter for powerline noise (50 Hz)
        print(f"Applying notch filter: {notch_freq} Hz")
        self.filtered_data.notch_filter(freqs=notch_freq, method='iir', verbose=False)
        
        print("Filtering completed")
        
    def plot_raw_vs_filtered(self, channel='EEG Fp1', duration=5):
        """Compare raw vs filtered data"""
        if self.raw_data is None or self.filtered_data is None:
            print("Data not available for plotting")
            return
            
        # Get data for the specified channel
        raw_channel_data = self.raw_data.copy().pick([channel])
        filtered_channel_data = self.filtered_data.copy().pick([channel])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # Raw data
        start, stop = 0, int(duration * self.raw_data.info['sfreq'])
        times = self.raw_data.times[start:stop]
        raw_signal = raw_channel_data.get_data()[0, start:stop]
        
        ax1.plot(times, raw_signal, 'b-', linewidth=1, label='Raw')
        ax1.set_title(f'Raw EEG Signal - {channel}')
        ax1.set_ylabel('Amplitude (µV)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Filtered data
        filtered_signal = filtered_channel_data.get_data()[0, start:stop]
        
        ax2.plot(times, filtered_signal, 'r-', linewidth=1, label='Filtered')
        ax2.set_title(f'Filtered EEG Signal - {channel}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (µV)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def remove_artifacts_ica(self, n_components=15):
        """Remove artifacts using ICA"""
        if self.filtered_data is None:
            print("No filtered data available")
            return
            
        print("\nREMOVING ARTIFACTS WITH ICA")
        print("="*50)
        
        try:
            # Create copy for ICA
            ica_data = self.filtered_data.copy()
            
            # Set up ICA
            ica = mne.preprocessing.ICA(n_components=n_components, 
                                      random_state=42,
                                      method='fastica')
            
            print("Fitting ICA...")
            ica.fit(ica_data)
            
            # Use the existing ECG channel for artifact detection
            artifact_components = []
            
            # Find ECG artifacts using the real ECG channel
            if 'ECG ECG' in ica_data.ch_names:
                print("Using real ECG channel for artifact detection")
                ecg_indices, ecg_scores = ica.find_bads_ecg(ica_data, 
                                                           method='correlation',
                                                           threshold=0.8,
                                                           ch_name='ECG ECG')
                if ecg_indices:
                    artifact_components.extend(ecg_indices)
                    print(f"Found {len(ecg_indices)} ECG artifact components")
            
            # Find EOG artifacts using frontal channels
            eog_indices, eog_scores = ica.find_bads_eog(ica_data, threshold=2.0)
            if eog_indices:
                artifact_components.extend(eog_indices)
                print(f"Found {len(eog_indices)} EOG artifact components")
            
            # Remove duplicates
            artifact_components = list(set(artifact_components))
            
            print(f"Identified {len(artifact_components)} total artifact components: {artifact_components}")
            
            if artifact_components:
                # Apply ICA to remove artifacts
                ica.exclude = artifact_components
                self.cleaned_data = ica_data.copy()
                ica.apply(self.cleaned_data)
                print("Artifact removal completed")
            else:
                print("No artifacts detected, using filtered data")
                self.cleaned_data = self.filtered_data.copy()
                
        except Exception as e:
            print(f"ICA failed: {e}")
            print("Using filtered data as cleaned data")
            self.cleaned_data = self.filtered_data.copy()
    
    def plot_psd_comparison(self):
        """Plot Power Spectral Density before and after processing"""
        if self.raw_data is None or self.cleaned_data is None:
            print("Data not available for PSD comparison")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Raw PSD
        self.raw_data.compute_psd(fmax=50).plot(axes=ax1, show=False)
        ax1.set_title('Power Spectrum - Raw Data')
        
        # Cleaned PSD
        self.cleaned_data.compute_psd(fmax=50).plot(axes=ax2, show=False)
        ax2.set_title('Power Spectrum - Cleaned Data')
        
        plt.tight_layout()
        plt.show()
    
    def save_processed_data(self, output_dir='processed_data'):
        """Save the processed EEG data"""
        if self.cleaned_data is None:
            print("No cleaned data to save")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        if hasattr(self.raw_data, 'filenames') and self.raw_data.filenames:
            original_name = os.path.basename(self.raw_data.filenames[0])
            output_name = os.path.splitext(original_name)[0] + '_processed.fif'
        else:
            output_name = 'eeg_processed.fif'
        
        output_path = os.path.join(output_dir, output_name)
        
        # Save the data
        self.cleaned_data.save(output_path, overwrite=True)
        print(f"Processed data saved to: {output_path}")
        return True
        
    def run_full_pipeline(self, edf_file_path, save_output=True):
        """Run the complete preprocessing pipeline"""
        print("STARTING EEG PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load data
        if not self.load_edf_file(edf_file_path):
            return False
            
        # 2. Show basic info
        self.basic_info()
        
        # 3. Apply filtering
        self.apply_filtering()
        
        # 4. Plot comparison
        self.plot_raw_vs_filtered(channel='EEG Fp1', duration=5)
        
        # 5. Remove artifacts with ICA
        self.remove_artifacts_ica()
        
        # 6. Plot PSD comparison
        self.plot_psd_comparison()
        
        # 7. Save processed data
        if save_output:
            self.save_processed_data()
        
        print("\nPREPROCESSING PIPELINE COMPLETED!")
        return True


# Simple test function
def preprocess_all():
    """Run preprocessing on all EDF files in the dataset"""
    # dataset_path = r"C:\Users\nevin\OneDrive\Desktop\EEGMAT_dataset\eeg-during-mental-arithmetic-tasks-1.0.0"
    dataset_path = r"C:\Users\aroraak\OneDrive - Umich\umich fall 25\EECS 598 BIO AI\EEGMAT_dataset-main\EEGMAT_dataset-main\eeg-during-mental-arithmetic-tasks-1.0.0"
    
    # Collect all EDF files recursively
    edf_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    
    if not edf_files:
        print(" No EDF files found in the dataset directory")
        return
    
    print(f" Found {len(edf_files)} EDF files to process\n")
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor()
    
    for i, file_path in enumerate(edf_files, start=1):
        print(f"\n================= FILE {i}/{len(edf_files)} =================")
        print(f"Processing: {os.path.basename(file_path)}")
        print("===========================================")
        success = preprocessor.run_full_pipeline(file_path)
        if not success:
            print(f" Skipped {os.path.basename(file_path)} due to an error.")
    
    print("\n ALL FILES PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    preprocess_all()
