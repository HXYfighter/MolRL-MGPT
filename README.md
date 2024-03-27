# MolRL-MGPT
This is the code repository for our paper published in NeurIPS 2023: [De novo Drug Design using Reinforcement Learning with Multiple GPT Agents](https://arxiv.org/abs/2401.06155).

## Dependency

```bash
pytorch==1.12.1

```

### Dataset

Following MolBART, we use a filtered ZINC dataset containing 100M SMILES. The files are available at [https://github.com/MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer).

The filtered ChEMBL dataset is available at 



## Pre-training

```bash
python codes/pretrain.py 
```

## MARL

### GuacaMol benchmark

```bash
python codes/MARL.py --task_id 0
```

### SARS-COV-2 protein targets

```bash
python codes/MARL.py --oracle docking_PLPro_7JIR_mpo
python codes/MARL.py --oracle docking_RdRp_mpo
```

### 