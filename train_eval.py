#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 3 16:14:52 2025

@author: sielviesharma
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def prepare_dataloaders(df):
    X_ids = torch.stack(list(df['bert_input'])).long()
    X_mask = torch.stack(list(df['bert_mask'])).long()
    X_aug = torch.tensor(df['aug'].tolist(), dtype=torch.float32)
    y = torch.tensor(df['label'].tolist(), dtype=torch.long)

    X_train_ids, X_test_ids, X_train_mask, X_test_mask, X_train_aux, X_test_aux, y_train, y_test = train_test_split(
        X_ids, X_mask, X_aug, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train_ids, X_train_mask, X_train_aux, y_train)
    test_dataset = TensorDataset(X_test_ids, X_test_mask, X_test_aux, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader, y_test

def train_model(model, train_loader, device, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, aux_batch, y_batch in train_loader:
            input_ids, attention_mask, aux_batch, y_batch = (
                input_ids.to(device), attention_mask.to(device),
                aux_batch.to(device), y_batch.to(device))

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, aux_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, y_test, label_encoder, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, attention_mask, aux_batch, y_batch in test_loader:
            input_ids, attention_mask, aux_batch = (
                input_ids.to(device), attention_mask.to(device), aux_batch.to(device))
            outputs = model(input_ids, attention_mask, aux_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    print("\nBlaDe Model Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
