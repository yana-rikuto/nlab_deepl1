# main.py

from src.data_preprocessing import preprocess_data
from src.feature_extraction import extract_features
from src.model_training import build_model, train_model, save_model
from src.evaluation import evaluate_model, load_trained_model

# データの前処理
train_data, train_labels, test_data, test_labels = preprocess_data()

# 特徴量の抽出
train_features = extract_features(train_data)
test_features = extract_features(test_data)

# モデルの構築と訓練
input_shape = (train_features.shape[1], train_features.shape[2])
model = build_model(input_shape)
train_model(model, train_features, train_labels)

# モデルの保存
model_filepath = 'trained_model.h5'
save_model(model, model_filepath)

# 訓練されたモデルのロードと評価
trained_model = load_trained_model(model_filepath)
evaluate_model(trained_model, test_features, test_labels)
