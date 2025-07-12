#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 5 13:14:02 2025

@author: sielviesharma
"""

from transformers import BertTokenizer
import torch
from blade_model import blademodel
from data_loader import load_and_preprocess_data, extract_aux
from train_eval import prepare_dataloaders, evaluate_model

def predict_lang(model, tokenizer, sentence, device, label_encoder):
    model.eval()
    encoded = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    aug = torch.tensor([extract_aux(sentence)], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask, aug)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

def test_and_predict():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    df, label_encoder = load_and_preprocess_data(tokenizer)
    _, test_loader, y_test = prepare_dataloaders(df)

    model = blademodel(hidden_dim=64, aug_dim=5, output_dim=len(label_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

   
    model.load_state_dict(torch.load("blade_model.pt"))

    evaluate_model(model, test_loader, y_test, label_encoder, device)
    print("\nSample Prediction:")
    sentence = "Bonjour tout le monde"
    pred = predict_lang(model, tokenizer, sentence, device, label_encoder)
    print(f"Sentence: '{sentence}' --> Predicted Language: {pred}")


    # print("\nSample Predictions:")
    # sample_sentences ="Bonjour tout le monde";
    # for sentence in sample_sentences:
    #     pred = predict_lang(model, tokenizer, sentence, device, label_encoder)
    #     print(f"Sentence: '{sentence}' --> Predicted Language: {pred}")

if __name__ == '__main__':
    test_and_predict()
