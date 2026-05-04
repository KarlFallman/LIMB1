#pytorch
#Parametrar XYZ, ID
#INPUT HIDDEN EXPORT ONNX
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
HIDDEN_SIZE = 64
EMBED_SIZE = 32
MAX_SEQ_LEN = 26  # Max längd för padding
BATCH_SIZE = 32
EPOCHS = 10
EXPORT_THRESHOLD = 0.01
ONNX_PATH = "movement_gru.onnx"


# =========================
# FRAME -> VECTOR
# =========================
def frame_to_vector(frame):
    vec = []

    for joint in ["shoulder", "elbow", "wrist"]:
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
    sequence_num = raw["sequence"]  # Används kanske senare
    data = raw["data"]

    seq = [frame_to_vector(f) for f in data]
    seq = np.stack(seq)

    return seq, user_id, sequence_num


# =========================
# SPLIT INTO SEQUENCES
# =========================
def split_sequence(seq, seq_len):
    seq_len = min(seq_len, MAX_SEQ_LEN)  # Begränsa till max
    if len(seq) < seq_len:
        # Om sekvensen är kortare, använd hela och pad till seq_len
        pad_len = seq_len - len(seq)
        pad = np.zeros((pad_len, seq.shape[1]))
        seq_padded = np.vstack([seq, pad])
        return [seq_padded]
    
    chunks = []
    for i in range(0, len(seq) - seq_len + 1, seq_len):
        chunk = seq[i:i+seq_len]
        # Pad chunk till MAX_SEQ_LEN om nödvändigt
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
    def __init__(self, folder):
        self.samples = []

        for file in os.listdir(folder):
            if not file.endswith(".json"):
                continue

            seq, user_id, sequence_num = load_sequence(os.path.join(folder, file))
            chunks = split_sequence(seq, sequence_num)

            for c in chunks:
                self.samples.append((c, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, user_id = self.samples[idx]
        return seq, user_id


# =========================
# TRIPLET DATASET
# =========================
class TripletDataset(Dataset):
    def __init__(self, base):
        self.data = base.samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, user = self.data[idx]

        positives = [x for x in self.data if x[1] == user]
        negatives = [x for x in self.data if x[1] != user]

        positive = random.choice(positives)[0]
        negative = random.choice(negatives)[0]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )


# =========================
# MODEL (GRU)
# =========================
class MovementGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, EMBED_SIZE)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        emb = self.fc(out)
        return F.normalize(emb, dim=1)


def export_model_to_onnx(model, path):
    model.eval()
    dummy_input = torch.randn(1, MAX_SEQ_LEN, INPUT_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {path}")


# =========================
# TRAIN LOOP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MovementGRU().to(device)
if os.path.exists("movement_gru.pth"):
    model.load_state_dict(torch.load("movement_gru.pth"))
    print("Loaded existing model.")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.TripletMarginLoss(margin=1.0)

dataset = TripletDataset(MovementDataset("data"))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = [dataset[j] for j in range(i, min(i+BATCH_SIZE, len(dataset)))]

        anchor = torch.stack([b[0] for b in batch]).to(device)
        positive = torch.stack([b[1] for b in batch]).to(device)
        negative = torch.stack([b[2] for b in batch]).to(device)

        a = model(anchor)
        p = model(positive)
        n = model(negative)

        loss = criterion(a, p, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    average_loss = total_loss / num_batches if num_batches else total_loss
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Avg: {average_loss:.4f}")

    if average_loss < EXPORT_THRESHOLD:
        export_model_to_onnx(model, ONNX_PATH)
        break


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "movement_gru.pth")