import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import random
import os
import csv
import math
import h5py
import re
from glob import glob
from tqdm import tqdm


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow_addons.optimizers import AdamW

from ..src.metrics import calculate_eer, compute_tdcf
from ..src.model.lcnn import build_lcnn
from .feature_extraction import _calc_stft, _extract_label

# === Model Config ===
epochs = 50
batch_size = 32
lr = 0.001
feature_type = "stft"
input_shape = (257, 200, 1)

# === Dataset Paths ===
protocol_tr = r"D:\mine\Shsh\HIAST\projects\grad project\codes\LCNN-master\my_model\src\protocol\train_protocol.csv"
protocol_dev = r"D:\mine\Shsh\HIAST\projects\grad project\codes\LCNN-master\my_model\src\protocol\dev_protocol.csv"
protocol_eval = r"D:\mine\Shsh\HIAST\projects\grad project\codes\LCNN-master\my_model\src\protocol\eval_protocol.csv"

path_tr = r"D:/mine/Shsh/HIAST/projects/grad project/dataset/LA/ASVspoof2019_LA_train/flac/"
path_dev = r"D:/mine/Shsh/HIAST/projects/grad project/dataset/LA/ASVspoof2019_LA_dev/flac/"
path_eval = r"D:/mine/Shsh/HIAST/projects/grad project/dataset/LA/ASVspoof2019_LA_eval/flac/"

# === Feature Paths ===
base_feature_path = f'D:/mine/Shsh/HIAST/projects/grad project/codes/LCNN-master/my_model/features/{feature_type}/'
x_train_path = os.path.join(base_feature_path, f'x_train_{feature_type}.npy')
y_train_path = os.path.join(base_feature_path, f'y_train_{feature_type}.npy')
x_val_path = os.path.join(base_feature_path, f'x_val_{feature_type}.npy')
y_val_path = os.path.join(base_feature_path, f'y_val_{feature_type}.npy')

# === Load or Extract Features ===
if feature_type == "stft":
    if all(map(os.path.exists, [x_train_path, x_val_path, y_train_path, y_val_path])):
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
    else:
        print("Extracting train data...")
        df_tr = pd.read_csv(protocol_tr)
        x_train, y_train = calc_stft(df_tr, path_tr)
        print("Extracting dev data...")
        df_dev = pd.read_csv(protocol_dev)
        x_val, y_val = calc_stft(df_dev, path_dev)
        print("Saving features...")
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_val_path, x_val)
        np.save(y_val_path, y_val)

# === Weights ===
weights_dir = f"D:/mine/Shsh/HIAST/projects/grad project/codes/LCNN-master/my_model/weights/{feature_type}/"
os.makedirs(weights_dir, exist_ok=True)

# Find latest saved weights
latest_epoch = 0
latest_weights_path = None
weights_files = glob(os.path.join(weights_dir, "model_epoch_*.weights.h5"))
for wf in weights_files:
    match = re.search(r"model_epoch_(\d+)\.weights\.h5", os.path.basename(wf))
    if match:
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_weights_path = wf

# === Model Setup ===
lcnn = build_lcnn(input_shape)
if latest_weights_path and os.path.exists(latest_weights_path):
    lcnn.load_weights(latest_weights_path)
    print(f"✅ Loaded weights from: {latest_weights_path}")
    initial_epoch = latest_epoch
else:
    print("❌ No weights found, starting from scratch.")
    initial_epoch = 0

lcnn.compile(
    optimizer=AdamW(learning_rate=lr, weight_decay=1e-04),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# === Callbacks for Training ===
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self.epochs_per_save = kwargs.pop('epochs_per_save', 1)
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epochs_per_save == 0:
            self.filepath = os.path.join(weights_dir, f"model_epoch_{epoch + 1}.weights.h5")
            super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)

def scheduler(epoch, lr):
    return lr if epoch < 2 else lr * math.exp(-0.1)

lrs = LearningRateScheduler(scheduler)
es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
cp_cb = CustomModelCheckpoint(
    filepath=os.path.join(weights_dir, f"model_epoch_{initial_epoch + 1}.weights.h5"),
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    epochs_per_save=1
)

# === Train if Needed ===
log_path = os.path.join(base_feature_path, 'training_metrics.csv')
def log_metrics(history, filename):
    if not history or not hasattr(history, 'history'):
        print("ℹ️ No history to log (training was skipped or failed).")
        return
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        num_epochs = len(history.history.get('loss', []))
        for epoch_index in range(num_epochs):
            row = {
                'epoch': initial_epoch + epoch_index + 1,
                'accuracy': history.history.get('accuracy')[epoch_index],
                'loss': history.history.get('loss')[epoch_index],
                'val_accuracy': history.history.get('val_accuracy')[epoch_index],
                'val_loss': history.history.get('val_loss')[epoch_index],
            }
            writer.writerow(row)

if initial_epoch < epochs:
    print(f"▶️ Resuming training from epoch {initial_epoch + 1} to {epochs}")
    history = lcnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[es, cp_cb, lrs],
        initial_epoch=initial_epoch
    )
    log_metrics(history, log_path)
else:
    print("✅ All epochs already completed. Skipping training.")

# === Evaluation ===
print("\n🔎 Evaluating model...")

# === Efficient Evaluation ===
print("\n🔎 Evaluating model in batches...")

def eval_generator(protocol_df, path, batch_size):
    labels = _extract_label(protocol_df)
    for i in range(0, len(protocol_df), batch_size):
        batch_files = protocol_df["utt_id"].iloc[i:i+batch_size]
        batch_data = []
        for audio_id in batch_files:
            file = os.path.join(path, audio_id + ".flac")
            stft = _calc_stft(file)  
            batch_data.append(stft)
        x_batch = np.stack(batch_data, axis=0)
        yield x_batch, labels[i:i+batch_size]


df_eval = pd.read_csv(protocol_eval)
eval_steps = math.ceil(len(df_eval) / batch_size)
gen = eval_generator(df_eval, path_eval, batch_size)

all_scores = []
all_labels = []

for x_batch, y_batch in tqdm(gen, total=eval_steps):
    preds = lcnn.predict(x_batch, batch_size=batch_size, verbose=0)
    all_scores.extend(preds[:, 0])
    all_labels.extend(y_batch)

# Convert to numpy arrays
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# === Metrics
eer = calculate_eer(all_labels, all_scores)
print(f"📊 EER: {eer * 100:.2f}%")

p_target = 0.05
c_miss = 1
c_fa = 10
tdcf, thresholds = compute_tdcf(all_labels, all_scores, p_target, c_miss, c_fa)
min_tdcf = np.min(tdcf)
min_tdcf_threshold = thresholds[np.argmin(tdcf)]

print(f"📉 min t-DCF: {min_tdcf:.4f}")
print(f"📈 Threshold for min t-DCF: {min_tdcf_threshold:.4f}")


print(f"📊 EER: {eer * 100:.2f}%")
