DIR_RAW = 'data/converted'
DIR_FILTERED = 'data/filtered_data'
DIR_ICA_FILTERED = 'data/ica_filtered_data'

LABELS_PATH = 'data/labels.csv'

COLUMNS_TO_RENAME = {
    'Subject No.': 'subject_no',
    'Trial_1': 't1_math',
    'Unnamed: 2': 't1_mirror',
    'Unnamed: 3': 't1_stroop',
    'Trial_2': 't2_math',
    'Unnamed: 5': 't2_mirror',
    'Unnamed: 6': 't2_stroop',
    'Trial_3': 't3_math',
    'Unnamed: 8': 't3_mirror',
    'Unnamed: 9': 't3_stroop'
}

DATA_TYPES = ["raw", "filtered", "ica_filtered"]


TEST_TYPES = ["background", "task"] 

TEST_TYPE_COLUMNS = {
    "background": ["_1"],  # suffix “_1” indicates background recording
    "task": ["_2"]         # suffix “_2” indicates during arithmetic task
}

N_CLASSES = 2
SFREQ = 500 # sampling frequency
