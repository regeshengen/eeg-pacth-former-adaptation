import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pickle
from google.colab import drive

drive.mount('/content/drive')

base_path = '/content/drive/MyDrive/EEG_data/' 
output_path_corrected = '/content/drive/MyDrive/EEG_data/eeg_data.pkl' 

# --- Funções de Filtragem ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0: 
        b, a = butter(order, high, btype='low', analog=False)
        print(f"Applying low-pass filter with highcut={highcut}Hz as lowcut was <=0")
    elif high >= 1: 
        b, a = butter(order, low, btype='high', analog=False)
        print(f"Applying high-pass filter with lowcut={lowcut}Hz as highcut was >=fs/2")
    else:
        b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, data, axis=0)
    return y

def notch_filter(data, notch_freq, quality_factor, fs):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    y = filtfilt(b, a, data, axis=0)
    return y
# --- Fim Funções de Filtragem ---


def load_and_filter_eeg_data(filepath, fs_known, lowcut=1.0, highcut=40.0, notch_f=50.0, quality_f=30):
    mat_data = loadmat(filepath, squeeze_me=True, struct_as_record=False)

    main_data_struct = None
    possible_struct_keys = ['cnt', 'EEG', 'data']
    found_key = None

    for key_candidate in possible_struct_keys:
        if key_candidate in mat_data:
            found_key = key_candidate
            break
    if not found_key:
        for key in mat_data.keys():
            if not key.startswith('__'):
                found_key = key
                break
    if not found_key:
        print(f"ERROR: Could not determine main data structure in {filepath}. Skipping.")
        return None, None, None

    main_data_struct = mat_data[found_key]
    eeg_signals, channel_names, sampling_frequency = None, None, None

    if hasattr(main_data_struct, 'x'):
        eeg_signals = main_data_struct.x
    else:
        print(f"ERROR: Field 'x' not found in {filepath}. Skipping.")
        return None, None, None

    if hasattr(main_data_struct, 'clab'):
        channel_names = main_data_struct.clab
    if hasattr(main_data_struct, 'fs'):
        sampling_frequency = main_data_struct.fs
        if sampling_frequency != fs_known:
            print(f"Warning: File {filepath} reports fs={sampling_frequency}, but processing with fs_known={fs_known}.")
    else:
        print(f"Warning: Field 'fs' not found in {filepath}. Using fs_known={fs_known}.")
        sampling_frequency = fs_known # Usar o fs conhecido se não estiver no arquivo

    if not isinstance(eeg_signals, np.ndarray) or eeg_signals.ndim != 2:
        print(f"ERROR: eeg_signals not a 2D array in {filepath}. Skipping.")
        return None, None, None
    
    # --- Aplicar Filtros ---
    print(f"  Original data shape: {eeg_signals.shape}, fs from file: {sampling_frequency if hasattr(main_data_struct, 'fs') else 'Not in file, used known'}")
    
    filtered_eeg = np.zeros_like(eeg_signals, dtype=np.float32)
    for i in range(eeg_signals.shape[1]): 
        channel_data = eeg_signals[:, i].astype(np.float64) 
        
        if notch_f > 0 and notch_f < fs_known/2:
            channel_data = notch_filter(channel_data, notch_f, quality_f, fs_known)

        channel_data = butter_bandpass_filter(channel_data, lowcut, highcut, fs_known, order=4)
        
        filtered_eeg[:, i] = channel_data.astype(np.float32)
    
    print(f"  Filtered data shape: {filtered_eeg.shape}")
    return filtered_eeg, channel_names, sampling_frequency 


eeg_data_list_filtered = [] 
participants = sorted([p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p)) and p.startswith('VP')])


FS_KNOWN = 200.0 
LOW_CUT = 1.0
HIGH_CUT = 40.0
NOTCH_FREQ = 50.0 
QUALITY_FACTOR = 30.0

print(f"Starting preprocessing with Fs={FS_KNOWN}Hz, Bandpass={LOW_CUT}-{HIGH_CUT}Hz, Notch={NOTCH_FREQ}Hz")

for participant in participants:
    participant_path = os.path.join(base_path, participant)
    
    files_in_participant_dir = os.listdir(participant_path)
    eeg_mat_files = sorted([f for f in files_in_participant_dir if f.endswith('.mat') and f.startswith('cnt_')])
    print(f"\nFound files for {participant}: {eeg_mat_files}")

    if not eeg_mat_files:
        print(f"No 'cnt_*.mat' files found for participant {participant}. Skipping.")
        continue

    for f_name in eeg_mat_files:
        task_name_from_file = f_name.replace('cnt_', '').split('.')[0].split('_')[0]
        file_path = os.path.join(participant_path, f_name)
        
        filtered_eeg_data, channels, fs_original = load_and_filter_eeg_data(
            file_path, 
            fs_known=FS_KNOWN, 
            lowcut=LOW_CUT, 
            highcut=HIGH_CUT, 
            notch_f=NOTCH_FREQ, 
            quality_f=QUALITY_FACTOR
        )
        
        if filtered_eeg_data is not None:
            eeg_data_list_filtered.append({
                'participant': participant,
                'task': task_name_from_file,
                'data': filtered_eeg_data, 
                'channels_names': channels, 
                'fs': fs_original if fs_original is not None else FS_KNOWN 
            })
        else:
            print(f"Skipping file {file_path} due to loading/filtering errors.")

with open(output_path_corrected, 'wb') as f:
    pickle.dump(eeg_data_list_filtered, f)
print(f'\nDataset filtrado gerado e salvo em: {output_path_corrected}')

if eeg_data_list_filtered:
    print("\nInspecting first item of the new PKL file:")
    first_item = eeg_data_list_filtered[0]
    print(f"Participant: {first_item['participant']}")
    print(f"Task: {first_item['task']}")
    print(f"Data type: {type(first_item['data'])}")
    print(f"Data dtype: {first_item['data'].dtype}")
    print(f"Data shape: {first_item['data'].shape}")
    print(f"Fs stored: {first_item.get('fs')}")
    if first_item.get('channels_names') is not None:
        print(f"Num Channels from clab: {len(first_item.get('channels_names', []))}")
    
    print(f"Data min: {np.min(first_item['data'])}, max: {np.max(first_item['data'])}, mean: {np.mean(first_item['data'])}")

else:
    print("\nNew PKL file is empty. Check loading logs.")