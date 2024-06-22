# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:10:22 2024

@author: Paulo
"""

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Diretório contendo os arquivos de áudio organizados por pastas de etiquetas
data_dir = 'C:/Users/Paulo/OneDrive/Área de Trabalho/TCC 2/choros sem ruido'

# Função para extrair MFCCs de um arquivo de áudio
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None

# Verificar se o diretório existe
if not os.path.exists(data_dir):
    print(f"O diretório especificado não existe: {data_dir}")
else:
    # Criação do dataset
    data = []
    labels = []
    max_pad_len = 400

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                mfccs = extract_mfcc(file_path, max_pad_len=max_pad_len)
                if mfccs is not None:
                    data.append(mfccs)
                    labels.append(label)

    # Verifique se os dados foram extraídos corretamente
    if len(data) == 0:
        print("Nenhum dado foi extraído. Verifique seus arquivos de áudio e o caminho do diretório.")
    else:
        # Conversão para DataFrame do pandas
        X = np.array(data)
        y = np.array(labels)

        # Verifique se os arrays não estão vazios
        if X.size == 0 or y.size == 0:
            print("Os arrays de dados ou etiquetas estão vazios. Verifique o processo de extração.")
        else:
            # Reshape para (n_samples, n_features)
            X = X.reshape(X.shape[0], -1)

            # Dividir os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Treinamento do modelo Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predição e avaliação do modelo
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Imprimir a precisão e as predições
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print("Predictions:")
            for i, prediction in enumerate(y_pred):
                print(f"Audio {i + 1}: Predicted={prediction}, Actual={y_test[i]}")

            # Função para prever a classe de um novo arquivo de áudio
            def predict_new_audio(file_path):
                mfccs = extract_mfcc(file_path, max_pad_len=max_pad_len)
                if mfccs is not None:
                    mfccs = mfccs.reshape(1, -1)  # Reshape para (1, n_features)
                    prediction = model.predict(mfccs)
                    return prediction[0]
                else:
                    print("Erro ao extrair MFCCs do novo áudio.")
                    return None

            # Solicitar ao usuário para fornecer o caminho do novo arquivo de áudio
            new_audio_path = input("Por favor, insira o caminho do novo arquivo de áudio: ")
            if os.path.exists(new_audio_path):
                prediction = predict_new_audio(new_audio_path)
                if prediction is not None:
                    print(f"A previsão para o novo áudio é: {prediction}")
            else:
                print(f"O arquivo especificado não existe: {new_audio_path}")
