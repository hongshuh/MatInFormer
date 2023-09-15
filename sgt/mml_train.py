from data_mb import Matbench_dataset,collate_batch
from model import SpaceGroupTransformer
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader,Subset
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from sklearn.metrics import mean_absolute_error
from utils_my import roberta_base_AdamW_LLRD
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torchsummary import summary
from tqdm import tqdm
def load_pretrained_model(model,pretrained_model):
    load_state = torch.load(pretrained_model) 
    model_state = model.state_dict()
    # print(model_state)
    # exit()
    for name, param in load_state.items():
        # print(name)
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

    
    config = yaml.load(open("config_mb.yaml", "r"), Loader=yaml.FullLoader)
    device = config['device']
    epochs = config['epochs']
    dataset_name = config['dataset_name']
    fold_num = config['fold_num']
    print(fold_num)
    pretrain_model = config['pretrain_model']
    torch.manual_seed(42)
    scaler = StandardScaler()
    train_data = Matbench_dataset(config,scaler=scaler,is_train=True)
    test_data  = Matbench_dataset(config,scaler=scaler,is_train=False)

    val_size = int(0.25 * len(train_data))
    validation_indices = torch.randperm(n=val_size).tolist()
    # print(validation_indices)
    # exit()
    validation_data = Subset(train_data,validation_indices)
    train_loader = DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,num_workers=4)
    val_loader = DataLoader(validation_data,batch_size=config['batch_size'],shuffle=True,num_workers=4)
    test_loader = DataLoader(test_data,batch_size=config['batch_size'],shuffle=False,num_workers=4)

    print(config)
    #Set up models
    model = SpaceGroupTransformer(config)
    model = model.to(device)
    model = torch.compile(model)

    if pretrain_model is not None:
        model = load_pretrained_model(model,pretrain_model)

    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {num_params}")
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
        for tokens_id,com_embed,mask_id,target in tqdm(train_loader):
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
        print(f'Epoch {i} : Train Loss {loss.item()}')
        with torch.no_grad():
            loss_val_all = 0.0
            for tokens_id,com_embed,mask_id,target in val_loader:
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
                best_path = f'models/best{fold_num}_{dataset_name}.pth'
                torch.save(model.state_dict(),best_path)
                print(f"Epoch {i} : MAE = {loss_val_all/len(validation_data)} better model than before")
            else:
                patience += 1
                print(f'Epoch {i} : MAE = {loss_val_all/len(validation_data)} not improve {patience} epoch')
            #TODO Saving the best model
        # exit()
    with torch.no_grad():
        loss_val_all = 0.0
        target_list = []
        pred_list = []
        model = load_pretrained_model(model,best_path)
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
    df.to_csv(f'results/test_results_{dataset_name}_{fold_num}.csv')
    print(f'MAE of test results_{dataset_name}_{fold_num}',mean_absolute_error(df['Pred'],df['GT']))

            


