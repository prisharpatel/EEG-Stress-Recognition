import os
import numpy as np
import pandas as pd
import scipy
import variablesEEGMAT as v # this is where they are getting the test types and data types from

def load_dataset(data_type="raw", test_type="task"):
    '''
    Loads data from the EEG Physionet Dataset.
    
    Args:
        data_type (string): The data type to load. Defaults to "ica_filtered".
        test_type (string): The test type to load. Defaults to "Arithmetic".
    
    Returns:
        ndarray: The specified dataset.
    # NOTE: subtract eeg background from eeg task data
        

    '''
    assert (test_type in v.TEST_TYPES)

    assert (data_type in v.DATA_TYPES)
    
    # if data_type == "ica_filtered" and test_type != "Arithmetic":
    #     print("Data of type", data_type, "does not have test type", test_type)
    #     return 0
    print("hi!!")
    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    
        
    dataset = []

    counter = 0
    for filename in os.listdir(dir):
        # go through for loop only if looking at backgroudn or task
        if test_type == "background" and not f.endswith("_1.edf"):
            continue
        if test_type == "task" and not f.endswith("_2.edf"):
            continue

        full_path = os.path.join(dir, f)
        print(f"Loading {f}...")
        mat = scipy.io.loadmat(full_path)
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[key]

        dataset.append(data)

        counter += 1

    dataset = np.array(dataset)
    print(f"Loaded {dataset.shape[0]} recordings from {dir} (shape per trial: {dataset.shape[1:]})")
    return dataset


def load_labels():
    '''
   TODO: Loads labels from the dataset and transforms the label values to binary values.

    Returns:
        ndarray: The labels.
    '''
    labels = pd.read_csv(v.LABELS_PATH)

    if 'Count quality' in labels.columns:
        label_array = labels['Count quality'].to_numpy().astype(int)

    print(f"Loaded {len(label_array)} labels (0=bad, 1=good)")
    return label_array


def format_labels(labels, test_type="task", epochs=1):
    '''
    Filter the labels and repeat for the specified amount of epochs.

    Args:
        labels (ndarray): The labels.
        test_type (string): The test_type to filter by. Defaults to "Arithmetic".
        epochs (int): The amount of epochs. Defaults to 1.

    Returns:
        ndarray: The formatted labels.

    '''
    assert test_type in v.TEST_TYPES

    return np.repeat(labels, epochs)


def split_data(data, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec.

    Args:
        data (ndarray): EEG data.
        sfreq (int): The sampling frequency.
    
    Returns:
        ndarray: The epoched data.

    '''
    n_trials, n_channels, n_samples = data.shape
    n_epochs = n_samples // sfreq
    epoched = np.empty((n_trials, n_epochs, n_channels, sfreq))

    for i in range(n_trials):
        for j in range(n_epochs):
            epoched[i, j] = data[i, :, j*sfreq:(j+1)*sfreq]

    print(f"Split into {n_epochs} epochs per trial → shape: {epoched.shape}")
    return epoched
