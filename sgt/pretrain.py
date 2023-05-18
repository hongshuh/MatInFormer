# from data import Matbench_dataset,collate_batch
import wandb
from data_pretrain import Pretrain_data
from model_pretrain import SpaceGroupTransformer
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

    
    config = yaml.load(open("config_pretrain.yaml", "r"), Loader=yaml.FullLoader)
    device = config['device']
    epochs = config['epochs']
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = current_time
    log_dir = os.path.join('runs_contrast', dir_name)
    config['log_dir'] = log_dir
    os.makedirs(log_dir,exist_ok=True)
    wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config=config
    
)


    data = Pretrain_data(config)
    val_ratio = 0.05
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size
    train_data, test_data = torch.utils.data.random_split(data,[train_size,val_size])
    train_loader = DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,num_workers=4)
    test_loader = DataLoader(test_data,batch_size=config['batch_size'],shuffle=False,num_workers=4)


    #Set up models
    model = SpaceGroupTransformer(config)
    model = model.to(device)

    model = torch.compile(model)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    # optimizer = roberta_base_AdamW_LLRD(model,config['lr'],config['weight_decay'])
    optimizer = torch.optim.AdamW(model.parameters(),config['lr'],weight_decay=config['weight_decay'])

    #Scheduler
    steps_per_epoch = math.ceil(len(train_data) // config['batch_size'])
    training_steps = steps_per_epoch * config['epochs']
    warmup_steps = int(training_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
    

    best_validation_score = float('inf')
    for i in range(epochs):
        batch_num = 0
        for tokens_id,com_embed,mask_id,target in train_loader:
            model.train()
            tokens_id = tokens_id.to(device)
            com_embed = com_embed.to(device)
            target = target.to(device)
            mask_id = mask_id.to(device)
            optimizer.zero_grad()
            outputs = model(tokens_id,com_embed,mask_id)
            loss = loss_fn(outputs.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_num % 100 == 0:
                print(f'Epoch {i} : Train Loss {loss.item()}')
            batch_num +=1
            wandb.log({"train_loss": loss})

        with torch.no_grad():
            loss_val_all = 0.0
            for tokens_id,com_embed,mask_id,target in test_loader:
                model.eval()
                tokens_id = tokens_id.to(device)
                com_embed = com_embed.to(device)
                target = target.to(device)
                mask_id = mask_id.to(device)

                outputs = model(tokens_id,com_embed,mask_id)
                loss_val_all += nn.MSELoss(reduction='sum')(target.squeeze(),outputs.squeeze())
            
            if loss_val_all < best_validation_score:
                best_validation_score = loss_val_all
                patience = 0
                # torch.save(model)if current_score <= best_score:
                best_path = f'{log_dir}/best.pth'
                torch.save(model.state_dict(),best_path)
                print(f"Epoch {i} : MSE = {loss_val_all/len(test_data)} better model than before")
            else:
                patience += 1
                print(f'Epoch {i} : MSE = {loss_val_all/len(test_data)} not improve {patience} epoch')
            #TODO Saving the best model
            checkpoint_path = f'{log_dir}/checkpoint.pth'
            torch.save(model.state_dict(),checkpoint_path)
        wandb.log({"val_metrics": loss_val_all/len(test_data)})
        
        # exit()
    with torch.no_grad():
        loss_val_all = 0.0
        target_list = []
        pred_list = []
        for tokens_id,com_embed,mask_id,target in test_loader:
            model.eval()
            tokens_id = tokens_id.to(device)
            com_embed = com_embed.to(device)
            mask_id = mask_id.to(device)
            outputs = model(tokens_id,com_embed,mask_id)
            pred_list += outputs.squeeze().detach().cpu().numpy().tolist()
            target_list += target.squeeze().numpy().tolist()

            


