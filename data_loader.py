#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 11:43:02 2025

@author: sielviesharma
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

MAX_LANGUAGES = 5

def locationofthefile():
    dataset_p = ["/Users/sielviesharma/Desktop/BLaDe/wili.csv"]
    for path in dataset_p:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate dataset CSV file.")

def augmented_fe(text):
    words = word_tokenize(text)
    num_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    num_digits = sum(c.isdigit() for c in text)
    num_puncts = sum(c in "!?.,;:" for c in text)
    stopword_r = sum(w in stop_words for w in words) / num_words if num_words else 0
    return [num_words, avg_word_len, num_digits, num_puncts, stopword_r]

def load_and_process_data(tokenizer, max_len=50):
    file_path = locationofthefile()
    df = pd.read_csv(file_path)
    top_languages = df['Language'].value_counts().nlargest(MAX_LANGUAGES).index.tolist()
    df = df[df['Language'].isin(top_languages)]
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Language'])

    def encode_bert(text):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    df['bert_input'] = df['Text'].apply(lambda x: encode_bert(x)[0])
    df['bert_mask'] = df['Text'].apply(lambda x: encode_bert(x)[1])
    df['aug'] = df['Text'].apply(augmented_fe)

    return df, le
