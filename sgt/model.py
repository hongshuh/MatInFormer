import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW,RobertaModel,RobertaConfig
from copy import deepcopy

class SpaceGroupTransformer(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        hidden_size = config['hidden_size'] 
        num_attention_heads = config['num_attention_heads']                 
        num_hidden_layers = config['num_hidden_layers']
        self.element_len = config['max_element']
        self.sequence_len = config['blocksize']
        self.max_position = config['max_position_embeddings']
        self.task = config['task']
        self.tokens_len = config['blocksize'] - self.element_len
        self.composition_word_embedding = nn.Linear(201,hidden_size)
        # self.word_embedding = nn.Embedding(780,hidden_size)
        # self.word_embedding = nn.Embedding(860,hidden_size)

        self.word_embedding = nn.Embedding(626,hidden_size)

        roberta_config = RobertaConfig(
            vocab_size=800, #800,1200,1500
            max_position_embeddings=self.max_position,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=1,
            hidden_size=hidden_size,
            hidden_dropout_prob=config['hidden_dropout_prob'],
            attention_probs_dropout_prob=config['attention_probs_dropout_prob']
        )
        self.transformer = RobertaModel(config=roberta_config)
        self.sigmoid = nn.Sigmoid()
        self.Regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self,tokens_id,com_embed,mask_id):
        # tokens_id = torch.tensor(tokens_id,dtype=int)
        tokens_embed = self.word_embedding(tokens_id.type(torch.int))
        compo_embed = self.composition_word_embedding(com_embed)
        # print(tokens_embed.shape)
        # print(compo_embed.shape)
        # exit()
        initial_embeddings = torch.cat([tokens_embed,compo_embed],axis = 1)
        # initial_embeddings = initial_embeddings.unsqueeze(0)
        # # position_id = torch.range(2,self.tokens_len+1) + torch.ones(self.element_len)
        # position_id = torch.cat([torch.arange(2,self.tokens_len+2,device=initial_embeddings.device)
        #                          ,-1*torch.ones(self.element_len,device=initial_embeddings.device)])
        # # position_id = position_id.to(device)
        # position_id = position_id.unsqueeze(0).expand(self.sequence_len)
        # print(position_id)
        # print(initial_embeddings.shape)
        outputs = self.transformer.forward(attention_mask = mask_id,inputs_embeds=initial_embeddings)

        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output