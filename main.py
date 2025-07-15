#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sielviesharma
"""


from transformers import BertTokenizer
import torch
from blade_model import blademodel
from data_loader import load_and_process_data
from train_eval import prepare_dataloaders, train_model, evaluate_model

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    df, label_encoder = load_and_process_data(tokenizer)

    train_loader, test_loader, y_test = prepare_dataloaders(df)

    model = blademodel(hidden_dim=64, aug_dim=5, output_dim=len(label_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, device, epochs=5)
    evaluate_model(model, test_loader, y_test, label_encoder, device)
    torch.save(model.state_dict(), "bisat_model.pt")


if __name__ == '__main__':
    main()
