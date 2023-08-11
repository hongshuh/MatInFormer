import numpy as np
import pandas as pd
import json
from pymatgen.core import Composition,Structure
from pymatgen.symmetry.groups import SpaceGroup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from porE.hea.HEA import HEA
from pore import porosity as p
from pore import psd 
from porE.io.ase2pore import *
import os, os.path

df=pd.read_csv('MOF_all.csv')
def run_GPA_fullgrid(xyz,probe_R,grid_a,grid_b,grid_c):
    '''
        Execute porosity evaluation, using the grid point apporach (GPA)
        Here: Provide an explicit size of the grid in directions a, b, c

        A probe radius for the accessible porosity needs to be provided

        Input
        -----
        xyz     ... structural information in a xyz format compatible with porE
        probe_R ... probe radius, in A
        grid_a  ... number of grid points in crystallographic direction a
        grid_b  ... number of grid points in crystallographic direction b
        grid_c  ... number of grid points in crystallographic direction c

        Output
        ------
        Phi_void   ... Void porosity, in %
        Phi_acc    ... Accessible porosity, in %
        density    ... density of the structure, in kg/m^3
        poreV_void ... pore volume density wrt void porosity, in cm^3/g
        poreV_acc  ... pore volume density wrt accessible porosity, in cm^3/g
    '''

    # print('')
    # print('-------------------------------')
    # print('Run GPA: grid_a, grid_b, grid_c')
    # print('-------------------------------')
    Phi_void, Phi_acc, density, poreV_void, poreV_acc = p.gpa_fullgrid(xyz,probe_R,grid_a,grid_b,grid_c)
    return Phi_void, Phi_acc, density, poreV_void, poreV_acc

N_points = 200
N_steps  = 2000
probe_R      = 1.20
grid_density = 5
void_list = []
acc_list = []
for idx in tqdm(range(len(df))):
    cif_id = df.iloc[idx]['id']
    xyz = f'./hMOF_xyz/{cif_id}.xyz'
    Phi_void, Phi_acc, density, poreV_void, poreV_acc = run_GPA_fullgrid(xyz,probe_R,grid_density,grid_density,grid_density)
    # # break
    void_list.append(Phi_void)
    acc_list.append(Phi_acc)
df['void'] = void_list
df['acc_void'] = acc_list
df.to_csv('MOF_pore.csv',index=None)