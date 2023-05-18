from pymatgen.core import Structure,Composition
from pymatgen.symmetry.groups import SpaceGroup

import torch
import numpy as np
import pandas as pd
import re
import json
import yaml
from matbench import MatbenchBenchmark

from utils_my import get_spg_wkf_tokens,get_composition_embedding,get_token_id
from torch.utils.data import Dataset,DataLoader

class SpaceGroupDataset(Dataset):
    def __init__(self,config,path,scaler
        ,is_train = True,
        **kwargs) -> None:
        self.is_train = is_train
        self.max_seq_len = config['blocksize']
        self.max_num_elem = 20
        self.scaler = scaler
        self.df = pd.read_csv(path)
        if is_train:
            self.target = 'energy_above_hull'
            self.df[self.target]=self.scaler.fit_transform(self.df[self.target].values.reshape(-1,1))
        else:
            self.target = 'e_above_hull_mp2020_corrected_ppd_mp'
            self.df[self.target]=self.scaler.transform(self.df[self.target].values.reshape(-1,1))

        
        with open(config['vocab_path']) as file:
            self.vocab = json.load(file)
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):

        # Get structure
        formula = self.df['formula'][idx]
        space_group = self.df['spg_symbol'][idx]
        space_group = SpaceGroup(space_group)
        target = self.df[self.target][idx]
        # Map Space Group tokens and Composition Embeddings
        spg_wkf_tokens = get_spg_wkf_tokens(space_group.full_symbol)
        composition_embeddings = get_composition_embedding(formula)
        cls_spg_wkf_token = ['CLS'] + spg_wkf_tokens
       
        # Creat Mask for SG tokens
        tokens_mask = np.zeros(self.max_seq_len - self.max_num_elem)
        tokens_mask[:len(cls_spg_wkf_token)] = 1
        # print(tokens_mask)

        #Padding SG Tokens
        if len(cls_spg_wkf_token) < self.max_seq_len - self.max_num_elem:
            pad_len = self.max_seq_len - self.max_num_elem - len(cls_spg_wkf_token)
            pad_token = ['PAD'] * pad_len
            cls_spg_wkf_token =cls_spg_wkf_token + pad_token


        # Creat Mask for Composition Embeddings and Paddding
        element_num = composition_embeddings.shape[0]
        comp_mask = np.zeros(self.max_num_elem)
        comp_mask[:element_num] = 1
        if element_num < self.max_num_elem:
            pad_embed = torch.zeros(self.max_num_elem-element_num,201)
            composition_embeddings = torch.cat((composition_embeddings,pad_embed))
        # print(comp_mask)
        cls_spg_wkf_token_id = get_token_id(cls_spg_wkf_token,self.vocab)


        # Convert Evetything into tensors
        cls_spg_wkf_token_id = torch.Tensor(cls_spg_wkf_token_id)
        composition_embeddings = torch.Tensor(composition_embeddings)
        target = torch.Tensor([target])
        mask_id = np.concatenate((tokens_mask,comp_mask),axis=None)
        mask_id = torch.Tensor(mask_id)
        # print(mask_id)
        return cls_spg_wkf_token_id,composition_embeddings,mask_id,target

def collate_batch(datalist):
    batch_tokens_id = []
    batch_com_embed = []
    batch_target = []
    batch_mask = []
    for i, (cls_spg_wkf_token_id, composition_embeddings, mask_id,target)in enumerate(datalist):
        batch_tokens_id.append(cls_spg_wkf_token_id)
        batch_com_embed.append(composition_embeddings)
        batch_target.append(target)
        batch_mask.append(mask_id)
    return torch.stack(batch_tokens_id),torch.stack(batch_com_embed),torch.stack(batch_mask),torch.stack(batch_target)



if __name__ == '__main__':
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

   