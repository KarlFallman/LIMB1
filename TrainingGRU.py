import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
INPUT_SIZE = 69
HIDDEN_SIZE = 128
EMBED_SIZE = 32
MAX_SEQ_LEN = 60

BATCH_SIZE = 8
EPOCHS = 500
LEARNING_RATE = 3e-4

VAL_SPLIT = 0.4
MIN_VAL_FILES = 2
PATIENCE = 75
EXPORT_THRESHOLD = 0.1
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
            chunks = split_sequence(seq, len(seq))

            for chunk in chunks:
                self.samples.append((chunk, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================
# TRIPLET DATASET (WITH SEMI-HARD MINING)
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

        # =========================
        # SEMI-HARD NEGATIVE MINING
        # =========================

        a = torch.tensor(anchor, dtype=torch.float32)
        p = torch.tensor(positive, dtype=torch.float32)

        anchor_pos_dist = torch.norm(a - p).item()

        semi_hard = []

        for neg in negatives:
            n = torch.tensor(neg[0], dtype=torch.float32)
            dist = torch.norm(a - n).item()

            if anchor_pos_dist < dist < anchor_pos_dist + 0.1:
                semi_hard.append(neg[0])

        if len(semi_hard) > 0:
            negative = random.choice(semi_hard)
        else:
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
        self.dropout = nn.Dropout(0.3)
        

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            EMBED_SIZE
        )

    def forward(self, x):
        out, _ = self.gru(x)
        #out = out.mean(dim=1)
        out = out[:, -1, :]
        out = self.dropout(out)
        emb = self.fc(out)
        return F.normalize(emb, dim=1)
        #return emb


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
            ratio = pos_dist / (neg_dist + 1e-8)

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

def compute_top1_accuracy(model, dataset, device):
    model.eval()

    correct = 0
    total = 0

    # bygg embeddings för alla users i val-set
    all_samples = dataset.data

    with torch.no_grad():
        for i in range(len(all_samples)):
            anchor, true_user = all_samples[i]

            anchor_tensor = torch.tensor(anchor, dtype=torch.float32).unsqueeze(0).to(device)
            anchor_emb = model(anchor_tensor).cpu().numpy()[0]

            best_user = None
            best_dist = float("inf")

            # jämför mot alla andra samples (enkelt men fungerar för din setup)
            for j in range(len(all_samples)):
                ref, ref_user = all_samples[j]

                ref_tensor = torch.tensor(ref, dtype=torch.float32).unsqueeze(0).to(device)
                ref_emb = model(ref_tensor).cpu().numpy()[0]

                dist = np.linalg.norm(anchor_emb - ref_emb)

                if dist < best_dist:
                    best_dist = dist
                    best_user = ref_user

            if best_user == true_user:
                correct += 1

            total += 1

    return correct / total
# =========================
# TRAIN
# =========================
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    folder = os.path.join(SCRIPT_DIR, "Data/Training")

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

    train_files = []
    val_files = []

    for user_id, user_list in user_files.items():
        random.shuffle(user_list)

        val_count = max(MIN_VAL_FILES, int(len(user_list) * VAL_SPLIT))

        val = user_list[:val_count]
        train = user_list[val_count:]

        train_files.extend(train)
        val_files.extend(val)

    train_dataset = TripletDataset(MovementDataset(folder, train_files))
    val_dataset = TripletDataset(MovementDataset(folder, val_files))

    model = MovementGRU().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
    criterion = nn.TripletMarginLoss(margin=0.7)

    best_val_loss = float("inf")
    patience_counter = 0
    onnx_exported = False

    for epoch in range(EPOCHS):

        train_loss, train_pos, train_neg = run_epoch(
            model, train_dataset, optimizer, criterion, device, True
        )

        val_loss, val_pos, val_neg = run_epoch(
            model, val_dataset, optimizer, criterion, device, False
        )
        val_acc = compute_top1_accuracy(model, val_dataset, device)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Pos: {val_pos:.4f} | "
            f"Val Neg: {val_neg:.4f} | "
            f"Val Ratio: {val_pos / (val_neg + 1e-8):.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pos = val_pos
            best_neg = val_neg
            best_ratio = val_pos / (val_neg + 1e-8)
            best_acc = val_acc
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

        else:
            patience_counter += 1
            print(f"No validation improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n===== FINAL BEST MODEL =====")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best positive distance: {best_pos:.4f}")
    print(f"Best negative distance: {best_neg:.4f}")
    print(f"Best ratio: {best_pos / (best_neg + 1e-8):.4f}")
    print(f"Best accuracy: {best_acc:.4f}")

    

if __name__ == "__main__":
    main()