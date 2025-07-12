#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 5 12:33:53 2025

@author: sielviesharma
"""
import torch
import torch.nn as nn
from transformers import BertModel

class blademodel(nn.Module):
    def __init__(self, hidden_dim, aug_dim, output_dim):
        super(blademodel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2 + aug_dim, output_dim)

    def forward(self, input_ids, attention_mask, aug):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(emb)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        combined = torch.cat((context, aug), dim=1)
        return self.fc(combined)
