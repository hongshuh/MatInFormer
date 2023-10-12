from pymatgen.core import Structure,Composition
from pymatgen.symmetry.groups import SpaceGroup

import torch
import numpy as np
import pandas as pd
import re
import json
import yaml
from matbench import MatbenchBenchmark

from utils_my import get_spg_wkf_tokens,get_composition_embedding,get_token_id,get_spg_tokens
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
class Matbench_dataset(Dataset):
    def __init__(self,config,scaler
        ,is_train = True,
        **kwargs) -> None:
        self.fold_num = config['fold_num']
        self.dataset_name = config['dataset_name']
        self.is_train = is_train
        self.max_seq_len = config['blocksize']
        self.max_num_elem = config['max_element']
        self.scaler=scaler

        mb = MatbenchBenchmark(autoload=False,subset=[self.dataset_name])
        for task in mb.tasks:
            task.load()
            if self.is_train:
                self.df = task.get_train_and_val_data(self.fold_num,as_type='df')
                self.df.iloc[:,1] = self.scaler.fit_transform(self.df.iloc[:,1].values.reshape(-1,1))
            else:
                self.df = task.get_test_data(self.fold_num,include_target=True,as_type='df')
                self.df.iloc[:,1] = self.scaler.transform(self.df.iloc[:,1].values.reshape(-1,1))

        with open(config['vocab_path']) as file:
            self.vocab = json.load(file)
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):

        # Get structure
        # print(self.df.iloc[idx])
        # exit()
        structure,target = self.df.iloc[idx]
        formula = structure.formula
        space_group = structure.get_space_group_info()[0]
        space_group = SpaceGroup(space_group)

        # Map Space Group tokens and Composition Embeddings
        # spg_wkf_tokens = get_spg_wkf_tokens(space_group.full_symbol)
        spg_wkf_tokens = get_spg_tokens(space_group.full_symbol)

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
    config = yaml.load(open("config_mb.yaml", "r"), Loader=yaml.FullLoader)

    # file = open('/home/hongshuh/matbench/matbench_jdft2d.json','r')
    # all_data = file.readlines()[0]
    # all_structure = json.loads(all_data)['data']
    # data = all_structure[10]
    # structure = Structure.from_dict(data[0])
    # target = data[1]


    mb_dataset = Matbench_dataset(config,is_train=True)
    print(mb_dataset.fold_num)
    print(mb_dataset.dataset_name)
    mb_dataloader = DataLoader(mb_dataset,batch_size=2,shuffle=True,collate_fn=collate_batch)
    # for tokens,comp_embed,target in mb_dataset:
    #     print(tokens.shape)
    #     print(comp_embed.shape)
    for tokens_id,com_embed,mask_id,target in mb_dataloader:
        exit()