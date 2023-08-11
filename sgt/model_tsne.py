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
        self.max_position = config['max_position_embeddings']
        vocab_size = config['vocab_size']
        self.composition_word_embedding = nn.Linear(201,hidden_size)
        self.word_embedding = nn.Embedding(626,hidden_size)
        roberta_config = RobertaConfig(
            vocab_size=800,
            max_position_embeddings=self.max_position,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.transformer = RobertaModel(config=roberta_config)


        self.linear = nn.Linear(hidden_size,vocab_size)
        
    def forward(self,tokens_id,com_embed,mask_id):
        tokens_id = torch.tensor(tokens_id,dtype=int)
        tokens_embed = self.word_embedding(tokens_id)
        compo_embed = self.composition_word_embedding(com_embed)
        # print(tokens_embed.shape)
        # print(compo_embed.shape)
        # exit()
        initial_embeddings = torch.cat([tokens_embed,compo_embed],axis = 1)
        # initial_embeddings = initial_embeddings.unsqueeze(0)
        
        # print(initial_embeddings.shape)
        outputs = self.transformer(attention_mask = mask_id,inputs_embeds=initial_embeddings,)

        logits = outputs.last_hidden_state[:, 0, :]
        output = logits
        

        return output