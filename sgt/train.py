from data_spg import SpaceGroupDataset
from model import SpaceGroupTransformer
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from sklearn.metrics import mean_absolute_error
from utils_my import roberta_base_AdamW_LLRD
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd
import wandb
def load_pretrained_model(model,pretrained_model):
    load_state = torch.load(pretrained_model) 
    model_state = model.state_dict()
    print(model_state)
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

    
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = config['device']
    epochs = config['epochs']
 
    pretrain_model = config['pretrain_model']
    scaler = StandardScaler()
    train_path = '/home/hongshuh/space_group_transformer/data/mp.csv'
    test_path = '/home/hongshuh/space_group_transformer/data/wbm.csv'
    wandb.init(
    # set the wandb project where this run will be logged
    project="fintune_WBM",
    
    # track hyperparameters and run metadata
    config=config
    
)
    train_data = SpaceGroupDataset(config,train_path,scaler=scaler,is_train=True)
    test_data = SpaceGroupDataset(config,test_path,scaler=scaler,is_train=False)

    train_loader = DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,num_workers=4)
    test_loader = DataLoader(test_data,batch_size=config['batch_size'],shuffle=False,num_workers=4)


    #Set up models
    model = SpaceGroupTransformer(config)
    model = model.to(device)
    model = torch.compile(model)

    if pretrain_model is not None:
        model = load_pretrained_model(model,pretrain_model)
    # exit()
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
        count = 0
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
            if count % 100 ==0:
                print(f'Epoch {i} : Train Loss {loss.item()}')
                count+=1
            wandb.log({'train_loss':loss.item()})
        with torch.no_grad():
            loss_val_all = 0.0
            for tokens_id,com_embed,mask_id,target in test_loader:
                model.eval()
                tokens_id = tokens_id.to(device)
                com_embed = com_embed.to(device)
                target = target.to(device)
                mask_id = mask_id.to(device)

                outputs = model(tokens_id,com_embed,mask_id)
                outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
                target = torch.from_numpy(scaler.inverse_transform(target.cpu().reshape(-1, 1)))
                loss_val_all += nn.L1Loss(reduction='sum')(target.squeeze(),outputs.squeeze())
            
            if loss_val_all < best_validation_score:
                best_validation_score = loss_val_all
                patience = 0
                # torch.save(model)if current_score <= best_score:
                best_path = f'wbm_results/best.pth'
                torch.save(model.state_dict(),best_path)
                print(f"Epoch {i} : MAE = {loss_val_all/len(test_data)} better model than before")
            else:
                patience += 1
                print(f'Epoch {i} : MAE = {loss_val_all/len(test_data)} not improve {patience} epoch')
            #TODO Saving the best model
            checkpoint_path = f'wbm_results/best.pth'
            torch.save(model.state_dict(),checkpoint_path)
            wandb.log({'val_metric':loss_val_all/len(test_data)})

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
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            target = torch.from_numpy(scaler.inverse_transform(target.cpu().reshape(-1, 1)))
            pred_list += outputs.squeeze().detach().cpu().numpy().tolist()
            target_list += target.squeeze().numpy().tolist()
    dict = {'Pred': pred_list, 'GT': target_list}
    df = pd.DataFrame(dict) 
    df.to_csv(f'wbm_results/test_results.csv')

            


