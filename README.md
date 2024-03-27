# MolRL-MGPT
This is the code repository for our paper published in NeurIPS 2023: [De novo Drug Design using Reinforcement Learning with Multiple GPT Agents](https://arxiv.org/abs/2401.06155).

## Dependencies

```bash
pytorch==1.12.1
rdkit==2020.03
tqdm
tensorboard
multiprocessing
PyTDC
openbabel
```

## Dataset & Docking

Following Chemformer, we use a filtered ZINC dataset containing 100M SMILES. The files are available at [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer).

The ChEMBL dataset is available at [ChEMBL](https://www.ebi.ac.uk/chembl/).

The SMILES vocabulary and protein structures can be found in `data/`.

Quick Vina 2 is available at [QuickVina](https://qvina.github.io/).

## Pre-training

```bash
python codes/pretrain.py 
```

## Multi-agent Reinforcement Learning

### GuacaMol benchmark

```bash
python codes/MARL.py --task_id 0
```

### SARS-COV-2 protein targets

```bash
python codes/MARL.py --oracle docking_PLPro_7JIR_mpo
python codes/MARL.py --oracle docking_RdRp_mpo
```

## Citation

```
@article{hu2024novo,
  title={De novo Drug Design using Reinforcement Learning with Multiple GPT Agents},
  author={Hu, Xiuyuan and Liu, Guoqing and Zhao, Yang and Zhang, Hao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

