import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


# =========================
# CONFIG
# =========================
INPUT_SIZE = 69
HIDDEN_SIZE = 128
EMBED_SIZE = 64
MAX_SEQ_LEN = 100

BATCH_SIZE = 64
EPOCHS = 250
LEARNING_RATE = 5e-4

VAL_SPLIT = 0.4
MIN_VAL_FILES = 2
PATIENCE = 50
EXPORT_THRESHOLD = 0.2
EXPORT_ONNX = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ONNX_PATH = os.path.join(SCRIPT_DIR, "movement_gru.onnx")
MODEL_PATH = os.path.join(SCRIPT_DIR, "movement_gru_best.pth")


# =========================
# FRAME -> VECTOR
# =========================
def frame_to_vector(frame):
    vec = []

    for joint in ["shoulder", "elbow"]:
        vec.extend(frame.get(joint, [0.0, 0.0, 0.0]))

    hand = frame.get("hand", [])
    hand_map = {p["id"]: p for p in hand}

    for i in range(21):
        if i in hand_map:
            p = hand_map[i]
            vec.extend([p["x"], p["y"], p["depth_m"]])
        else:
            vec.extend([0.0, 0.0, 0.0])

    return np.array(vec, dtype=np.float32)


# =========================
# LOAD JSON
# =========================
def load_sequence(path):
    with open(path, "r") as f:
        raw = json.load(f)

    user_id = raw["user_id"]
    sequence_num = raw["sequence"]
    data = raw["data"]

    seq = [frame_to_vector(frame) for frame in data]
    seq = np.stack(seq)

    return seq, user_id, sequence_num


# =========================
# SPLIT INTO CHUNKS
# =========================
def split_sequence(seq, seq_len):
    seq_len = min(seq_len, MAX_SEQ_LEN)

    if len(seq) < seq_len:
        pad_len = seq_len - len(seq)
        pad = np.zeros((pad_len, seq.shape[1]))
        seq = np.vstack([seq, pad])

    chunks = []

    for i in range(0, len(seq), seq_len):
        chunk = seq[i:i + seq_len]

        if len(chunk) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(chunk)
            pad = np.zeros((pad_len, seq.shape[1]))
            chunk = np.vstack([chunk, pad])

        chunks.append(chunk)

    return chunks


# =========================
# DATASET
# =========================
class MovementDataset(Dataset):
    def __init__(self, folder, files):
        self.samples = []

        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(folder, file)
            seq, user_id, sequence_num = load_sequence(path)
            chunks = split_sequence(seq, sequence_num)

            for chunk in chunks:
                self.samples.append((chunk, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================
# TRIPLET DATASET
# =========================
class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.data = base_dataset.samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, user = self.data[idx]

        positives = [
            x for x in self.data
            if x[1] == user and not np.array_equal(x[0], anchor)
        ]
        negatives = [
            x for x in self.data
            if x[1] != user
        ]

        positive = random.choice(positives)[0]
        negative = random.choice(negatives)[0]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )


# =========================
# MODEL
# =========================
class MovementGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            INPUT_SIZE,
            HIDDEN_SIZE,
            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            EMBED_SIZE
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        emb = self.fc(out)
        return F.normalize(emb, dim=1)


# =========================
# EXPORT ONNX
# =========================
def export_model_to_onnx(model, device):
    model.eval()

    dummy_input = torch.randn(
        1,
        MAX_SEQ_LEN,
        INPUT_SIZE,
        device=device
    )

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )

    print(f"Exported ONNX model -> {ONNX_PATH}")


# =========================
# RUN EPOCH
# =========================
def run_epoch(model, dataset, optimizer, criterion, device, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_pos = 0
    total_neg = 0
    num_batches = 0

    with torch.set_grad_enabled(training):
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = [
                dataset[j]
                for j in range(i, min(i + BATCH_SIZE, len(dataset)))
            ]

            anchor = torch.stack([b[0] for b in batch]).to(device)
            positive = torch.stack([b[1] for b in batch]).to(device)
            negative = torch.stack([b[2] for b in batch]).to(device)

            a = model(anchor)
            p = model(positive)
            n = model(negative)

            pos_dist = torch.norm(a - p, dim=1).mean().item()
            neg_dist = torch.norm(a - n, dim=1).mean().item()

            loss = criterion(a, p, n)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_pos += pos_dist
            total_neg += neg_dist
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_pos = total_pos / num_batches
    avg_neg = total_neg / num_batches

    return avg_loss, avg_pos, avg_neg


# =========================
# TRAIN
# =========================
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    folder = os.path.join(SCRIPT_DIR, "Data/Training")

    print("Current working directory:", os.getcwd())
    print("Script location:", SCRIPT_DIR)
    print("Folder path:", folder)
    print("Folder exists:", os.path.exists(folder))

    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    user_files = {}

    for file in files:
        path = os.path.join(folder, file)

        with open(path, "r") as f:
            raw = json.load(f)

        user_id = raw["user_id"]

        if user_id not in user_files:
            user_files[user_id] = []

        user_files[user_id].append(file)

    print("\nUsers found:")
    for user_id, user_list in user_files.items():
        print(user_id, len(user_list))

    train_files = []
    val_files = []

    for user_id, user_list in user_files.items():
        random.shuffle(user_list)

        val_count = max(
            MIN_VAL_FILES,
            int(len(user_list) * VAL_SPLIT)
        )

        if len(user_list) <= val_count:
            raise ValueError(
                f"User {user_id} has too few files ({len(user_list)})"
            )

        val = user_list[:val_count]
        train = user_list[val_count:]

        val_files.extend(val)
        train_files.extend(train)

        print(
            f"User {user_id}: "
            f"{len(train)} train / {len(val)} val"
        )

    print(f"Total train files: {len(train_files)}")
    print(f"Total val files: {len(val_files)}")

    train_dataset = TripletDataset(
        MovementDataset(folder, train_files)
    )

    val_dataset = TripletDataset(
        MovementDataset(folder, val_files)
    )

    model = MovementGRU().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.TripletMarginLoss(margin=0.5)

    best_val_loss = float("inf")
    best_pos = None
    best_neg = None

    patience_counter = 0
    onnx_exported = False

    for epoch in range(EPOCHS):
        train_loss, train_pos, train_neg = run_epoch(
            model,
            train_dataset,
            optimizer,
            criterion,
            device,
            training=True
        )

        val_loss, val_pos, val_neg = run_epoch(
            model,
            val_dataset,
            optimizer,
            criterion,
            device,
            training=False
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val Pos: {val_pos:.4f} | "
            f"Val Neg: {val_neg:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pos = val_pos
            best_neg = val_neg
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

            if (
                EXPORT_ONNX
                and val_loss < EXPORT_THRESHOLD
                and not onnx_exported
            ):
                export_model_to_onnx(model, device)
                onnx_exported = True

        else:
            patience_counter += 1
            print(
                f"No validation improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n===== FINAL BEST MODEL =====")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best positive distance: {best_pos:.4f}")
    print(f"Best negative distance: {best_neg:.4f}")


if __name__ == "__main__":
    main()