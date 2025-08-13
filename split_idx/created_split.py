import numpy as np
import os
# 3, 9, 16 is less
# 1, 8
# 分组
# 3，9，16，15
# 17， 18
# 1，12
# 2， 14
# 4， 5
# 6， 7
# 8，10
# 11， 13

split_dict = [{'train': [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14], 'test': [3, 9, 15, 16], 'val': [17, 18]},
              {'train': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16], 'test': [17, 18], 'val': [1, 12]},
              {'train': [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18], 'test': [1, 12], 'val': [2, 14]},
              {'train': [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18], 'test': [2, 14], 'val': [4, 5]},
              {'train': [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'test': [4, 5], 'val': [6, 7]},
              {'train': [1, 2, 3, 4, 5, 9, 11, 12, 13, 14, 15, 16, 17, 18], 'test': [6, 7], 'val': [8, 10]},
              {'train': [1, 2, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18], 'test': [8, 10], 'val': [11, 13]},
              {'train': [1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 17, 18], 'test': [11, 13], 'val': [3, 9, 15, 16]},
              ]

# check
for i in range(len(split_dict)):
    assert len(split_dict[i]['train']) + len(split_dict[i]['test']) + len(split_dict[i]['val']) == 18, \
        "Split {} does not contain all files.".format(i)
    assert len(set(split_dict[i]['train']) & set(split_dict[i]['test'])) == 0, \
        "Split {} contains overlap between train and test.".format(i)
    assert len(set(split_dict[i]['train']) & set(split_dict[i]['val'])) == 0, \
        "Split {} contains overlap between train and val.".format(i)
    assert len(set(split_dict[i]['test']) & set(split_dict[i]['val'])) == 0, \
        "Split {} contains overlap between test and val.".format(i)

npy_dir = 'split_idx'
np.save(os.path.join(npy_dir, 'idx_EFSleep.npy'), split_dict) 
print("Split indices saved to {}".format(os.path.join(npy_dir, 'idx_EFSleep.npy')))