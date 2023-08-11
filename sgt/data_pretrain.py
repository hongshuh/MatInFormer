from pymatgen.core import Structure,Composition
from pymatgen.symmetry.groups import SpaceGroup

import torch
import numpy as np
import pandas as pd
import re
import json
import yaml
from matbench import MatbenchBenchmark
import ast
from utils_my import get_spg_wkf_tokens,get_composition_embedding,get_token_id,get_spg_tokens
from torch.utils.data import Dataset,DataLoader

class Pretrain_data(Dataset):
    def __init__(self,config,
        **kwargs) -> None:
        self.max_seq_len = config['blocksize']
        self.json = json.load(open(config['path'],'r'))
        self.max_num_elem = config['max_element']
        # self.mask = config['mask']
        # if self.mask > 0:
        #     print(f'Mask {self.mask} tokens during pretrain')

        with open(config['vocab_path']) as file:
            self.vocab = json.load(file)
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.json['formula'])
    def __getitem__(self, idx):

        # Get structure
        # print(self.df.iloc[idx])
        # exit()
        # structure = Structure.from_dict(self.df.iloc[idx])
        # structure = self.df.iloc[idx][0]
        # print(structure)
        # exit()
        formula = self.json['formula'][idx]
        # print(formula)
        space_group = self.json['spg_symbol'][idx]
        # print(space_group)
        # exit()
        # # Map Space Group tokens and Composition Embeddings
        # spg_wkf_tokens = get_spg_wkf_tokens(space_group)
        spg_wkf_tokens = get_spg_tokens(space_group)

        composition_embeddings = get_composition_embedding(formula)
        cls_spg_wkf_token = ['CLS'] + spg_wkf_tokens
        # print(composition_embeddings.shape)
       
        # # Creat Mask for SG tokens
        tokens_mask = np.zeros(self.max_seq_len - self.max_num_elem)
        tokens_mask[:len(cls_spg_wkf_token)] = 1
        # # print(tokens_mask)

        # #Padding SG Tokens
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
        # print(cls_spg_wkf_token_id)

        # Mask some of the tokens
        # indices_mask = np.arange(1,len(cls_spg_wkf_token_id)) # Don't mask cls token
        # indices_mask = np.random.choice(indices_mask,3,replace=False)
        # tokens_mask[indices_mask] = 0
        # # print('indices_mask is',indices_mask)
        # # print('token id',cls_spg_wkf_token_id)

        # for idx in indices_mask:
        #     cls_spg_wkf_token_id[idx] = 625 # id of MASK tokens
        # print('MASK token id',cls_spg_wkf_token_id)
        # exit()
        # # Convert Evetything into tensors
        cls_spg_wkf_token_id = torch.Tensor(cls_spg_wkf_token_id)
        composition_embeddings = torch.Tensor(composition_embeddings)
        # target = torch.Tensor([target])
        mask_id = np.concatenate((tokens_mask,comp_mask),axis=None)
        mask_id = torch.Tensor(mask_id)
        # print(mask_id)
        a = self.json['a'][idx]
        b = self.json['b'][idx]
        c = self.json['c'][idx]
        alpha = self.json['alpha'][idx]
        beta = self.json['beta'][idx]
        gama = self.json['gama'][idx]
        target = [a,b,c,alpha,beta,gama]
        target = torch.Tensor(target)
        # exit()
        return cls_spg_wkf_token_id, composition_embeddings, mask_id,target

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
    config = yaml.load(open("config_pretrain.yaml", "r"), Loader=yaml.FullLoader)

    # file = open('/home/hongshuh/matbench/matbench_jdft2d.json','r')
    # all_data = file.readlines()[0]
    # all_structure = json.loads(all_data)['data']
    # data = all_structure[10]
    # structure = Structure.from_dict(data[0])
    # target = data[1]

    mb_dataset = Matbench_dataset_pretrain(config)
    print(len(mb_dataset))
    mb_dataloader = DataLoader(mb_dataset,batch_size=2,shuffle=True,collate_fn=collate_batch)
    # for tokens,comp_embed,target in mb_dataset:
    #     print(tokens.shape)
    #     print(comp_embed.shape)
    for items in mb_dataset:
        print(items)
        