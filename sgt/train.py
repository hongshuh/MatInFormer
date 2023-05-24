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
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
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
    train_path = config['train_data']
    test_path = config['test_data']
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
    # model = torch.compile(model)

    if pretrain_model is not None:
        model = load_pretrained_model(model,pretrain_model)
    # exit()
    if config['task'] == 'classification':
        loss_fn = nn.BCELoss()
        Validation_Metric = 'Accuracy'
    else:
        loss_fn = nn.L1Loss()
        Validation_Metric = 'MAE'
    # optimizer = roberta_base_AdamW_LLRD(model,config['lr'],config['weight_decay'])
    optimizer = torch.optim.AdamW(model.parameters(),config['lr'],weight_decay=config['weight_decay'])

    #Scheduler
    steps_per_epoch = math.ceil(len(train_data) // config['batch_size'])
    training_steps = steps_per_epoch * config['epochs']
    warmup_steps = int(training_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
    
    if config['task'] == 'classification':
        best_validation_score = 0.0
    else:
        best_validation_score = float('inf')
    for i in range(epochs):
        count = 0
        for tokens_id,com_embed,mask_id,target in train_loader:
            model.train()
            tokens_id = tokens_id.to(device)
            com_embed = com_embed.to(device)
            target = target.to(device)
            # if config['task'] == 'classification':
                # target = target.to(torch.int64)
            mask_id = mask_id.to(device)
            optimizer.zero_grad()
            outputs = model(tokens_id,com_embed,mask_id)
            loss = loss_fn(outputs.squeeze().float(), target.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            #test
            target = target.to(torch.int64)
            pred_label = outputs.squeeze().detach().cpu().numpy().round().tolist()
            target_label  = target.squeeze().cpu().numpy().tolist()
            # print('pred',pred_label[:10])
            # print('target',target_label[:10])
            # accuracy = accuracy_score(pred_label,target_label)
            # precision,recall,f1_score ,_ = precision_recall_fscore_support(pred_label,target_label)
            # print('accuracy',accuracy)
            # print('precision',precision)
            # print('recall',recall)
            # print('f1_score',f1_score)
            # exit()
            if count % 100 ==0:
                print(f'Epoch {i} : Train Loss {loss.item()}')
                count+=1
            wandb.log({'train_loss':loss.item()})
        if i % 5 == 0:
            with torch.no_grad():
                loss_val_all = 0.0
                pred_label_list=[]
                target_label_list=[]
                for tokens_id,com_embed,mask_id,target in test_loader:
                    model.eval()
                    tokens_id = tokens_id.to(device)
                    com_embed = com_embed.to(device)
                    target = target.to(device)
                    mask_id = mask_id.to(device)

                    outputs = model(tokens_id,com_embed,mask_id)
                    if config['task'] == 'classification':
                        target = target.to(torch.int64)
                        pred_label = outputs.squeeze().detach().cpu().numpy().round().tolist()
                        target_label  = target.squeeze().cpu().numpy().tolist()
                        target_label_list += target_label
                        pred_label_list += pred_label
                    else:
                        outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
                        target = torch.from_numpy(scaler.inverse_transform(target.cpu().reshape(-1, 1)))
                        loss_val_all += nn.L1Loss(reduction='sum')(target.squeeze(),outputs.squeeze())
                if config['task'] == 'classification':
                    precision,recall,f1_score,_=precision_recall_fscore_support(target_label_list,pred_label_list,average='binary')
                    accuracy = accuracy_score(target_label_list,pred_label_list)
                    loss_val_all = accuracy
                    if loss_val_all > best_validation_score:
                        best_validation_score = loss_val_all
                        patience = 0
                        # torch.save(model)if current_score <= best_score:
                        best_path = f'wbm_results/best.pth'
                        torch.save(model.state_dict(),best_path)
                        print(f"Epoch {i} : {Validation_Metric} = {accuracy} better model than before")
                    else:
                        patience += 1
                        print(f'Epoch {i} : {Validation_Metric} = {accuracy} not improve {patience} epoch')
                    #TODO Saving the best model
                    checkpoint_path = f'wbm_results/checkpoint.pth'
                    torch.save(model.state_dict(),checkpoint_path)
                    wandb.log({'precision':precision,'recall':recall,'f1_score':f1_score,'accuracy':accuracy})
                else:
                    if loss_val_all < best_validation_score:
                        best_validation_score = loss_val_all
                        patience = 0
                        # torch.save(model)if current_score <= best_score:
                        best_path = f'wbm_results/best.pth'
                        torch.save(model.state_dict(),best_path)
                        print(f"Epoch {i} : {Validation_Metric} = {loss_val_all/len(test_data)} better model than before")
                    else:
                        patience += 1
                        print(f'Epoch {i} : {Validation_Metric} = {loss_val_all/len(test_data)} not improve {patience} epoch')
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
            if config['task'] == 'classification':
                target = target.to(torch.int64)
                pred_label = outputs.squeeze().detach().cpu().numpy().round().tolist()
                target_label  = target.squeeze().cpu().numpy().tolist()
                target_list += target_label
                pred_list += pred_label
            else:
                outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
                target = torch.from_numpy(scaler.inverse_transform(target.cpu().reshape(-1, 1)))
                pred_list += outputs.squeeze().detach().cpu().numpy().tolist()
                target_list += target.squeeze().numpy().tolist()
    dict = {'Pred': pred_list, 'GT': target_list}
    df = pd.DataFrame(dict) 
    df.to_csv(f'wbm_results/test_results.csv')

            


