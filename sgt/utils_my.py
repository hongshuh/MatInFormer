from pymatgen.core import Structure,Composition
from pymatgen.symmetry.groups import SpaceGroup

import torch
import numpy as np
import pandas as pd
import re
import json
from os.path import abspath, dirname, join
import subprocess
from itertools import chain, groupby, permutations, product
from operator import itemgetter
from os.path import abspath, dirname, join
from shutil import which
from string import ascii_uppercase, digits
from typing import Literal

from monty.fractions import gcd
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch.optim import AdamW
module_dir = dirname(abspath(__file__))

with open(f'{module_dir}/embeddings/matscholar200.json') as file:
    elem_features = json.load(file)
# with open(f'{module_dir}/embeddings/wyckoff-position-multiplicities.json') as file:
#     # dictionary mapping Wyckoff letters in a given space group to their multiplicity
#     wyckoff_multiplicity_dict = json.load(file)

# with open(f'{module_dir}/embeddings/wyckoff-position-params.json') as file:
#     param_dict = json.load(file)
# with open(f'{module_dir}/embeddings/wyckoff-position-relabelings.json') as file:
#     relab_dict = json.load(file)
with open(f'{module_dir}/spg_dict.json') as file:
    spg_dict = json.load(file)
with open(f'{module_dir}/wkf_dict.json') as file:
    wkf_dict = json.load(file)



def get_spg_wkf_tokens(spg_symbol: str):
    spg_num = str(SpaceGroup(spg_symbol).int_number)
    spg_tokens = list(spg_dict[spg_symbol].values())
    wkf_tokens = wkf_dict[spg_num]
    return spg_tokens+wkf_tokens

def get_spg_tokens(spg_symbol: str):
    spg_num = str(SpaceGroup(spg_symbol).int_number)
    spg_tokens = list(spg_dict[spg_symbol].values())
    
    return spg_tokens

def get_token_id(tokens,vocab):
    tokens_id = []
    for token in tokens:
        # try:
        token = str(token)
        tokens_id.append(vocab[token])
        # except:
        #     print(tokens)
    return tokens_id
def get_composition_embedding(formula: str):
    """Concatenate matscholar element embeddings with element ratios in composition.

    Args:
        formula (str): Composition string.

    Returns:
        Tensor: Shape (n_elements, n_features). Usually (2-6, 200).
    """
    composition_dict = Composition(formula).get_el_amt_dict()
    elements, elem_weights = zip(*composition_dict.items())

    elem_weights = np.atleast_2d(elem_weights).T / sum(elem_weights)

    element_features = np.vstack([elem_features[el] for el in elements])

    # convert all data to tensors
    element_ratios = torch.tensor(elem_weights)
    element_features = torch.tensor(element_features)

    combined_features = torch.cat([element_ratios, element_features], dim=1).float()

    return combined_features

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)