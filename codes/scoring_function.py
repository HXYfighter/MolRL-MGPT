import os
import glob
import numpy as np
from tdc import Oracle, Evaluator

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import pybel

import subprocess
import multiprocessing


def int_div(smiles):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smiles)


def get_scores(smiles, mode="QED", n_process=16):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    pool = multiprocessing.Pool(processes = n_process)
    for index in range(n_process):
        temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, )))
    pool.close()
    pool.join()
    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()

    for filename in glob.glob("docking/mols/*"):
        if os.path.exists(filename):
            os.remove(filename)

    return scores

def get_scores_subproc(smiles, mode):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    if mode == "QED":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_QED([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "SA":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_SA([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "DRD2":
        oracle = Oracle(name='DRD2')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "GSK3B":
        oracle = Oracle(name='GSK3B')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "JNK3":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "docking_PLPro_7JIR":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                scores += [docking_score]
            else:
                scores += [-1.0]

    elif mode == "docking_PLPro_7JIR_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                scores += [0.8 * docking_score + 0.1 * oracle_QED(smiles[i]) + 0.1 * (10 - oracle_SA(smiles[i])) / 9]
                if docking_score > 0.5:
                    print(smiles[i], docking_score)
            else:
                scores += [-1.0]

    elif mode == "docking_RdRp":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[93.88, 83.08, 97.29])
                scores += [docking_score]
            else:
                scores += [-1.0]

    elif mode == "docking_RdRp_mpo":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/RDRP.pdbqt", box_center=[93.88, 83.08, 97.29])
                scores += [0.8 * docking_score + 0.1 * oracle_QED(smiles[i]) + 0.1 * (10 - oracle_SA(smiles[i])) / 9]
                if docking_score > 0.5:
                    print(smiles[i], docking_score)
            else:
                scores += [-1.0]

    else:
        raise Exception("Scoring function undefined!")


    return scores


def docking(smiles, receptor_file, box_center, box_size=[20, 20, 20]):
    if smiles == "":
        return -1.0

    ligand_mol_file = f"./docking/mols/mol_{smiles}.mol"
    ligand_pdbqt_file = f"./docking/mols/mol_{smiles}.pdbqt"
    docking_pdbqt_file = f"./docking/mols/dock_{smiles}.pdbqt"

    # 3D conformation of SMILES
    try:
        run_line = 'obabel -:%s --gen3D -O %s' % (smiles, ligand_mol_file)
        result = subprocess.check_output(run_line.split(), stderr=subprocess.STDOUT,
                    timeout=30, universal_newlines=True)
    except Exception as e:
        # print(e)
        return -1.0

    # docking by quick vina
    try:
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = 'docking/qvina02 --receptor %s --ligand %s --out %s' % (receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (box_center[0], box_center[1], box_center[2])
        run_line += ' --size_x %s --size_y %s --size_z %s' % (box_size[0], box_size[1], box_size[2])
        run_line += ' --cpu %d' % (4)
        run_line += ' --num_modes %d' % (10)
        run_line += ' --exhaustiveness %d ' % (4)
        result = subprocess.check_output(run_line.split(),
                                            stderr=subprocess.STDOUT,
                                            timeout=100, universal_newlines=True)
        result_lines = result.split('\n')
        affinity_list = list()
        check_result = False
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
            affinity_score = affinity_list[0]

        return reverse_sigmoid_transformation(affinity_score)

    except Exception as e:
        # print(e)
        return -1.0


def reverse_sigmoid_transformation(original_score): 
    if original_score > 99.9:
        return -1.0 
    else: # return (0, 1)
        _low = -12
        _high = -8
        _k = 0.25
        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = _reverse_sigmoid_formula(original_score, _low, _high, _k) 
        return transformed