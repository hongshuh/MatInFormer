# from data import Matbench_dataset,collate_batch
import wandb
from data_pretrain import Pretrain_data
from data_spg import SpaceGroupDataset
from model_tsne import SpaceGroupTransformer
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from sklearn.metrics import mean_absolute_error
from utils_my import roberta_base_AdamW_LLRD
import math
import os
import datetime
import pandas as pd
from tqdm import tqdm
import time
from sklearn.manifold import TSNE
import joblib
def load_pretrained_model(model,pretrained_model):
    load_state = torch.load(pretrained_model) 
    model_state = model.state_dict()
    # print(model_state)
    # exit()
    for name, param in load_state.items():
        print(name)
        if name not in model_state:
            print('NOT loaded:', name)
            continue
        else:
            print('loaded:', name)
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
        model_state[name].copy_(param)
        print("Loaded pre-trained model with success.")
    return model


if __name__ == '__main__':

    
    config = yaml.load(open("config_tsne.yaml", "r"), Loader=yaml.FullLoader)
    device = config['device']
    epochs = config['epochs']
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = current_time
    log_dir = os.path.join('tSNE')
    config['log_dir'] = log_dir
    os.makedirs(log_dir,exist_ok=True)


    pretrain_model = config['pretrain_model']
    data = Pretrain_data(config)
    # path = '/home/hongshuh/space_group_transformer/data/mp.csv'
    # data = SpaceGroupDataset(config,path,scaler='None')
    train_loader = DataLoader(data,batch_size=config['batch_size'],shuffle=False,num_workers=4)
    


    #Set up models
    model = SpaceGroupTransformer(config)
    model = model.to(device)

    model = torch.compile(model)
    # loss_fn = nn.MSELoss()

    pretrain_state = 'None'
    if pretrain_model is not None:
        model = load_pretrained_model(model,pretrain_model)
        pretrain_state = 'pretrained'
    # loss_fn = nn.MSELoss()
    # optimizer = roberta_base_AdamW_LLRD(model,config['lr'],config['weight_decay'])
    # optimizer = torch.optim.AdamW(model.parameters(),config['lr'],weight_decay=config['weight_decay'])

    #Scheduler
    # steps_per_epoch = math.ceil(len(train_data) // config['batch_size'])
    # training_steps = steps_per_epoch * config['epochs']
    # warmup_steps = int(training_steps * config['warmup_ratio'])
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                                     num_training_steps=training_steps)
    

    best_validation_score = float('inf')
    embeddings = []
    with torch.no_grad():
        model.eval()

        for tokens_id,com_embed,mask_id,target in tqdm(train_loader):
            tokens_id = tokens_id.to(device)
            com_embed = com_embed.to(device)
            target = target.to(device)
            mask_id = mask_id.to(device)
            outputs = model(tokens_id,com_embed,mask_id)
            embeddings.append(outputs.detach().cpu().numpy())
        # print('shape of output is',outputs.shape)
        # exit()
    embeddings = np.concatenate(embeddings,axis=0)
    print(embeddings.shape)
    # with open(f'{log_dir}/embeddings_mp.npy', 'wb') as f:
    #     np.save(f,embeddings)

    with open(f'{log_dir}/embeddings_{pretrain_state}.npy', 'wb') as f:
        np.save(f,embeddings)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)
    joblib.dump(tsne,f'tSNE/model_ocp')
    np.save(f"tSNE/tsne_ocp.npy",tsne_results)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

            


