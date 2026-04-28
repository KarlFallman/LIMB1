#pytorch,jupyter
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
#SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 10


# =========================
# FRAME -> VECTOR
# =========================
def frame_to_vector(frame):
    vec = []

    for joint in ["shoulder", "elbow", "wrist"]:
        vec.extend(frame.get(joint, [0.0, 0.0, 0.0]))

    hand = frame.get("hand", [])
    hand_map = {p["id"]: p for p in hand}

    for i in range(20):
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
    data = raw["data"]

    seq = [frame_to_vector(f) for f in data]
    seq = np.stack(seq)

    return seq, user_id


# =========================
# SPLIT INTO SEQUENCES
# =========================
def split_sequence(seq):
    chunks = []
    for i in range(0, len(seq) - SEQ_LEN + 1, SEQ_LEN):
        chunks.append(seq[i:i+SEQ_LEN])
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

            seq, user_id = load_sequence(os.path.join(folder, file))
            #chunks = split_sequence(seq)

            #for c in chunks:
            self.samples.append((c, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, user_id = self.samples[idx]
        return self.samples[idx]


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


# =========================
# TRAIN LOOP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MovementGRU().to(device)
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

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "movement_gru.pth")