# src/data_preprocessing.py

import numpy as np

def load_data():
    """
    データをロードする関数。
    事前に準備されたnumpy配列からデータを読み込む。
    """
    train_data = np.load('../data/train_data.npy')
    train_labels = np.load('../data/train_labels.npy')
    test_data = np.load('../data/test_data.npy')
    test_labels = np.load('../data/test_labels.npy')
    return train_data, train_labels, test_data, test_labels

def normalize_data(data):
    """
    データを正規化する関数。
    すべての値を0から1の範囲にスケールする。
    """
    return data / np.max(data)

def preprocess_data():
    """
    データの前処理を行うメイン関数。
    """
    train_data, train_labels, test_data, test_labels = load_data()
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)
    return train_data, train_labels, test_data, test_labels
