import os
from sklearn.model_selection import KFold
import glob
import numpy as np
import argparse
from sklearn.model_selection import StratifiedGroupKFold
def split_k_fold(dataset, num_splits, seed=666):
    
    expanded_labels = []
    for subject_id, stages in dataset:
        for stage, count in stages.items():
            expanded_labels.append((subject_id, stage, count))

    subjects = [entry[0] for entry in expanded_labels]
    labels = [entry[1] for entry in expanded_labels]
    label_counts = [entry[2] for entry in expanded_labels]

    sgkf = StratifiedGroupKFold(n_splits=num_splits, random_state=seed, shuffle=True)
    split_idx = []
    for train_idx, test_idx in sgkf.split(label_counts, labels, groups=subjects):
        train_set = [expanded_labels[i] for i in train_idx]
        test_set = [expanded_labels[i] for i in test_idx]

        train_label_counter = {"0":0, "1":0, "2":0, "3":0, "4":0}
        for _, label, count in train_set:
            train_label_counter[label] += count
        test_label_counter = {"0":0, "1":0, "2":0, "3":0, "4":0}
        for _, label, count in test_set:
            test_label_counter[label] += count
        
        train_set = list(set([entry[0] for entry in train_set]))
        test_set = list(set([entry[0] for entry in test_set]))
        split_idx.append({'train': train_set, 'test': test_set})

        print("Train distribution:", train_label_counter)
        print("Test distribution:", test_label_counter)
    for i in range(num_splits):
        val_set = split_idx[i-1]["test"]
        split_idx[i]["val"] = val_set
        train_set = split_idx[i]["train"]
        train_set = np.setdiff1d(train_set, val_set)
        split_idx[i]["train"] = train_set
        print(f"train: {len(split_idx[i]['train'])}, val: {len(split_idx[i]['val'])}, test: {len(split_idx[i]['test'])}")
        print(f"train: {split_idx[i]['train']}, val: {split_idx[i]['val']}, test: {split_idx[i]['test']}")
    return split_idx


def split_EFSleep(dset_name, num_splits, root_dir, eeg_channel, seed=666):
    np.random.seed(seed)
    dataset_path = os.path.join(root_dir, 'dset', dset_name, 'npz')
    data_root = os.path.join(dataset_path, eeg_channel)
    # print(data_root)
    data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
    subject_idx_list = [int(x[1:3]) for x in data_fname_list]
    subject_idx_list = np.unique(subject_idx_list)
    subject_idx_dict = [(idx, {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}) for idx in subject_idx_list]
    for i in range(len(data_fname_list)):
        for j in range(len(subject_idx_dict)):
            if int(data_fname_list[i][1:3]) == subject_idx_dict[j][0]:
                data = np.load(os.path.join(data_root, data_fname_list[i]), allow_pickle=True)['label']
                for k in range(len(data)):
                    subject_idx_dict[j][1][str(data[k])] += 1
    
    for i in range(len(subject_idx_dict)):
        print(f"Subject {subject_idx_dict[i][0]} distribution: {subject_idx_dict[i][1]}")
    split_idx = split_k_fold(subject_idx_dict, num_splits, seed)
    
    for i in range(num_splits):
        assert len(np.intersect1d(split_idx[i]['train'], split_idx[i]['val'])) == 0, f'Error: val_set and train_set have common elements in split {i}'
        assert len(np.intersect1d(split_idx[i]['train'], split_idx[i]['test'])) == 0, f'Error: test_set and train_set have common elements in split {i}'
        assert len(np.intersect1d(split_idx[i]['val'], split_idx[i]['test'])) == 0, f'Error: test_set and val_set have common elements in split {i}'
    
    np.save('split_idx//idx_{}.npy'.format(dset_name), np.array(split_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_name', type=str, default='EFSleep', help='Dataset name')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits')
    parser.add_argument('--root_dir', type=str, default='./', help='Root directory of the dataset')
    parser.add_argument('--eeg_channel', type=str, default='Fpz', help='EEG channel name')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    args = parser.parse_args()
    split_EFSleep(args.dset_name, args.num_splits, args.root_dir, args.eeg_channel, args.seed)