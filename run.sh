torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 0 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 1

torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 1 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 1 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 0 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_crl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_pretrain.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 0

# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 0 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 1

# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 1 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 1 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 1 --hb 1 --ppg 0 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_freezefinetune.json --gpu 0,1 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 0
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_scratch.json --gpu 3,4 --eeg 1 --hbo 1 --hb 1 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_scratch.json --gpu 3,4 --eeg 1 --hbo 0 --hb 0 --ppg 1 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_scratch.json --gpu 3,4 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 1
# torchrun --nproc_per_node=2 --master_port 12345 train_mtcl.py --config configs/SleePyCo-AgentTransformer_SL-10_numScales-3_EFSleep_scratch.json --gpu 3,4 --eeg 1 --hbo 0 --hb 0 --ppg 0 --eog 0