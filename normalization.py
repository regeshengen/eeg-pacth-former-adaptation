%%writefile dataset_tu_berlin.py

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

class TUBerlinDataset(Dataset):
    def __init__(self, pkl_file_path, participants_to_use, mode='train',
                 window_size=20,
                 step_size=5,  
                 label_map={'nback': 0, 'dsr': 1, 'wg': 2},
                 sampling_rate=200): 

        self.window_size = window_size
        self.step_size = step_size
        self.label_map = label_map
        self.mode = mode
        self.sampling_rate = sampling_rate

        print(f"[TUBerlinDataset {mode}] Loading data from: {pkl_file_path}")
        print(f"[TUBerlinDataset {mode}] Config: window_size={self.window_size}, step_size={self.step_size}, fs={self.sampling_rate} Hz")
        print(f"[TUBerlinDataset {mode}] Label map: {self.label_map}")

        with open(pkl_file_path, 'rb') as f:
            all_data_raw_list = pickle.load(f)

        self.data = []
        self.labels = []
        self.subject_ids = []
        num_channels_global = -1

        for item in all_data_raw_list:
            participant = item['participant']
            task_name = item['task']
            eeg_signal_from_pkl = item['data']

            if participant not in participants_to_use:
                continue

            if not isinstance(eeg_signal_from_pkl, np.ndarray) or eeg_signal_from_pkl.ndim != 2:
                # print(f"Warning: Data for p {participant}, t {task_name} is not 2D np.array. Shape: {type(eeg_signal_from_pkl)}. Skip.")
                continue

            current_channels = eeg_signal_from_pkl.shape[1]
            current_time_points = eeg_signal_from_pkl.shape[0]

            if num_channels_global == -1:
                num_channels_global = current_channels
                # print(f"[TUBerlinDataset {mode}] Detected {num_channels_global} channels from first valid file for participant {participant}.")
            elif num_channels_global != current_channels:
                # print(f"Warning: Inconsistent channel count for p {participant}, t {task_name}. Exp {num_channels_global}, got {current_channels}. Skip.")
                continue

            if current_time_points < self.window_size :
                continue

            if task_name not in self.label_map:
                # print(f"Warning: Unknown task '{task_name}' for p {participant}. Not in label_map. Skip.")
                continue

            label = self.label_map[task_name]

            eeg_signal_transposed = eeg_signal_from_pkl.T

            for i in range(0, current_time_points - self.window_size + 1, self.step_size):
                window = eeg_signal_transposed[:, i : i + self.window_size]
                self.data.append(window)
                self.labels.append(label)
                self.subject_ids.append(participant)

        if not self.data and participants_to_use:
             print(f"WARNING: [TUBerlinDataset {mode}] No data segments generated for participants {participants_to_use}. "
                   "Check if all files were too short for window_size or if other filters apply.")

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        if len(self.data) > 0:
            print(f"[TUBerlinDataset {mode}] Finished loading. Number of samples: {len(self.data)}")
            print(f"[TUBerlinDataset {mode}] Data shape of first sample: {self.data[0].shape}, Label of first sample: {self.labels[0]}")
        else:
            print(f"[TUBerlinDataset {mode}] Finished loading. Dataset is EMPTY for participants: {participants_to_use}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

    def get_num_channels(self):
        if len(self.data) > 0:
            return self.data[0].shape[0]
        # print("Warning: get_num_channels() called on an empty or uninitialized dataset, returning 0.")
        return 0

    def get_sequence_length(self):
        return self.window_size

def get_participant_splits(pkl_file_path, train_ratio=0.7, val_ratio=0.15, random_state=42):
    with open(pkl_file_path, 'rb') as f:
        all_data_raw = pickle.load(f)
    participants = sorted(list(set([item['participant'] for item in all_data_raw])))
    num_participants = len(participants)

    # print(f"[Splitter] Total unique participants found in pkl: {num_participants}. List: {participants}")

    if num_participants == 0:
        print("ERROR: [Splitter] No participants found in PKL file.")
        return [], [], []

    if num_participants < 2 and (train_ratio < 1.0 or val_ratio > 0 or (1.0-train_ratio-val_ratio)>0) :
        # print(f"Warning: [Splitter] Only {num_participants} participant(s). Cannot split. Assigning all to training.")
        return participants, [], []

    dummy_X = np.arange(num_participants)
    train_participants = []
    temp_participants_list = []

    temp_ratio_overall = round(1.0 - train_ratio, 4)
    if temp_ratio_overall <= 0:
        train_participants = participants
    elif temp_ratio_overall >= 1.0:
        temp_participants_list = participants
    else:
        n_temp_samples = int(round(temp_ratio_overall * num_participants))
        if n_temp_samples == 0 and num_participants > 1 and train_ratio < 1.0 : n_temp_samples = 1
        if n_temp_samples == num_participants and num_participants > 1 and train_ratio > 0.0 : n_temp_samples = num_participants - 1

        n_temp_samples = max(0, min(n_temp_samples, num_participants))
        if num_participants > 0 and n_temp_samples == num_participants and train_ratio > 0: # Ensure train is not empty
            n_temp_samples = num_participants -1
        if num_participants > 0 and n_temp_samples == 0 and temp_ratio_overall > 0: # Ensure temp is not empty if it should exist
             n_temp_samples = 1 if num_participants > 1 else 0


        if n_temp_samples == 0:
            train_participants = participants
        elif n_temp_samples == num_participants: # All to temp
            temp_participants_list = participants
            train_participants = []
        else:
            gss_train_temp = GroupShuffleSplit(n_splits=1, test_size=n_temp_samples, random_state=random_state)
            train_indices, temp_indices = next(gss_train_temp.split(dummy_X, groups=dummy_X))
            train_participants = [participants[i] for i in train_indices]
            temp_participants_list = [participants[i] for i in temp_indices]

    val_participants = []
    test_participants = []

    if not temp_participants_list:
        pass
    else:
        num_temp_participants = len(temp_participants_list)
        if num_temp_participants == 0:
             pass
        else:
            test_ratio_global = round(1.0 - train_ratio - val_ratio, 4)
            if test_ratio_global < 0: test_ratio_global = 0

            val_ratio_global = round(val_ratio, 4)
            if val_ratio_global < 0: val_ratio_global = 0

            if temp_ratio_overall <= 0:
                test_split_proportion_in_temp = 0
            else:
                test_split_proportion_in_temp = round(test_ratio_global / temp_ratio_overall, 4) if temp_ratio_overall > 0 else 0

            if test_split_proportion_in_temp <= 0:
                val_participants = temp_participants_list
            elif test_split_proportion_in_temp >= 1.0:
                test_participants = temp_participants_list
            elif num_temp_participants <= 1:
                if test_split_proportion_in_temp >= 0.5 :
                    test_participants = temp_participants_list
                else:
                    val_participants = temp_participants_list
            else:
                n_test_samples_in_temp = int(round(test_split_proportion_in_temp * num_temp_participants))

                if n_test_samples_in_temp == 0 and test_split_proportion_in_temp > 0 and num_temp_participants > 1 : n_test_samples_in_temp = 1
                if n_test_samples_in_temp == num_temp_participants and test_split_proportion_in_temp < 1 and num_temp_participants > 1: n_test_samples_in_temp = num_temp_participants - 1

                n_test_samples_in_temp = max(0, min(n_test_samples_in_temp, num_temp_participants))
                if num_temp_participants > 0 and n_test_samples_in_temp == num_temp_participants and test_split_proportion_in_temp < 1.0: # Ensure val is not empty
                    n_test_samples_in_temp = num_temp_participants -1
                if num_temp_participants > 0 and n_test_samples_in_temp == 0 and test_split_proportion_in_temp > 0: # Ensure test is not empty if it should exist
                    n_test_samples_in_temp = 1 if num_temp_participants > 1 else 0


                if n_test_samples_in_temp == 0:
                    val_participants = temp_participants_list
                elif n_test_samples_in_temp == num_temp_participants:
                    test_participants = temp_participants_list
                    val_participants = []
                else:
                    dummy_X_temp_split = np.arange(num_temp_participants)
                    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=n_test_samples_in_temp, random_state=random_state)
                    try:
                        val_indices_in_temp, test_indices_in_temp = next(gss_val_test.split(dummy_X_temp_split, groups=dummy_X_temp_split))
                        val_participants = [temp_participants_list[i] for i in val_indices_in_temp]
                        test_participants = [temp_participants_list[i] for i in test_indices_in_temp]
                    except ValueError as e:
                        # print(f"WARNING: [Splitter] ValueError during GSS split of temp list ({num_temp_participants} items, target test {n_test_samples_in_temp}): {e}")
                        # print("Fallback: Assigning all temp to validation.")
                        val_participants = temp_participants_list

    all_p_sets = [train_participants, val_participants, test_participants]
    set_names = ["Train", "Validation", "Test"]
    for i in range(len(all_p_sets)):
        for j in range(i + 1, len(all_p_sets)):
            if all_p_sets[i] and all_p_sets[j] and bool(set(all_p_sets[i]) & set(all_p_sets[j])):
                print(f"CRITICAL WARNING: [Splitter] Overlap detected between {set_names[i]} and {set_names[j]}!")
    return train_participants, val_participants, test_participants