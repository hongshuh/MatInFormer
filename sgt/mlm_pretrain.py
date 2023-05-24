# from data import Matbench_dataset,collate_batch
import wandb
from data_pretrain import Pretrain_data
from model_pretrain import SpaceGroupTransformer
import yaml
import torch
import numpy as np
from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
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

    
    config = yaml.load(open("config_pretrain_mask.yaml", "r"), Loader=yaml.FullLoader)
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
    project="MLM_pretrain",
    
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

    #Roberta config
    roberta_config = RobertaConfig(
        vocab_size=800,
        max_position_embeddings=config['max_position_embeddings'],
        num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers'],
        type_vocab_size=1,
        hidden_dropout_prob=config['hidden_dropout_prob'],
        attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
    )
    #Set up models
    model = RobertaForMaskedLM(config=roberta_config).to(device)
    model = torch.compile(model)
    # loss_fn = nn.MSELoss()
    
    training_args = TrainingArguments(
        output_dir=config['save_path'],
        overwrite_output_dir=config['overwrite_output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        save_strategy=config['save_strategy'],
        save_total_limit=config['save_total_limit'],
        fp16=config['fp16'],
        logging_strategy=config['logging_strategy'],
        evaluation_strategy=config['evaluation_strategy'],
        learning_rate=config['lr_rate'],
        lr_scheduler_type=config['scheduler_type'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        report_to=config['report_to'],
        dataloader_num_workers=config['dataloader_num_workers'],
        sharded_ddp=config['sharded_ddp'],
    )

    """Set Trainer"""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data
    )

    trainer.train(resume_from_checkpoint=config['load_checkpoint'])
    trainer.save_model(config["save_path"])