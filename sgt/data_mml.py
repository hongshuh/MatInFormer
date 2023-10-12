from pymatgen.core import Structure,Composition
from pymatgen.symmetry.groups import SpaceGroup

import torch
import numpy as np
import pandas as pd
import re
import json
import yaml
from matbench import MatbenchBenchmark
import warnings
from utils_my import get_spg_wkf_tokens,get_composition_embedding,get_token_id,get_spg_tokens
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
from torch_geometric import data

class Matbench_dataset(Dataset):
    def __init__(self,config,scaler
        ,is_train = True,max_num_nbr=12, radius=8, dmin=0, step=0.2,
        **kwargs) -> None:
        self.fold_num = config['fold_num']
        self.dataset_name = config['dataset_name']
        self.is_train = is_train
        self.max_seq_len = config['blocksize']
        self.max_num_elem = config['max_element']
        self.scaler=scaler
        self.radius = radius
        self.ari = AtomCustomJSONInitializer('./sgt/atom_init.json')
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
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

        atom_fea,nbr_fea,nbr_fea_idx = self.cgcnn_graph(structure)
        print(structure)
        print(f'atom fea {atom_fea.shape}')
        print(f'nbr fea {nbr_fea.shape}')
        print(f'nbr idx {nbr_fea_idx.shape}')
        exit()


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
    def cgcnn_graph(self,crystal):
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        # print(atom_fea.shape)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return atom_fea,nbr_fea,nbr_fea_idx
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


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


if __name__ == '__main__':
    config = yaml.load(open("config_mml.yaml", "r"), Loader=yaml.FullLoader)

    scaler = StandardScaler()
    mb_dataset = Matbench_dataset(config,is_train=True,scaler=scaler)

    train_loader = DataLoader(mb_dataset,batch_size=2)

    for cls_spg_wkf_token_id,composition_embeddings,mask_id,target in train_loader:
        print(1)
        break
  