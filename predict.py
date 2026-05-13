import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CONFIG
# =========================
INPUT_SIZE = 69
HIDDEN_SIZE = 128
EMBED_SIZE = 64
MAX_SEQ_LEN = 30

MODEL_PATH = "movement_gru_best.pth"

DATA_FOLDER = "Data/Training"   
TEST_FOLDER = "Test/Training"

TEST_FILE = "ID1test.json"   # filnamn inne i Test/


# =========================
# PREPROCESS
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


def load_sequence(path):
    with open(path, "r") as f:
        raw = json.load(f)

    data = raw["data"]

    seq = [frame_to_vector(frame) for frame in data]
    seq = np.stack(seq)

    return seq


def prepare_sequence(seq):
    seq = seq[:MAX_SEQ_LEN]

    if len(seq) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(seq)
        pad = np.zeros((pad_len, seq.shape[1]))
        seq = np.vstack([seq, pad])

    return seq


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
# EMBEDDING
# =========================
def get_embedding(model, file_path, device):
    seq = load_sequence(file_path)
    seq = prepare_sequence(seq)

    tensor = torch.tensor(
        seq,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.cpu().numpy()[0]


def distance(a, b):
    return np.linalg.norm(a - b)


# =========================
# BUILD REFERENCES
# =========================
def build_references(model, device):
    references = {}

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, file)

        with open(path, "r") as f:
            raw = json.load(f)

        user_id = raw["user_id"]

        if user_id not in references:
            references[user_id] = []

        references[user_id].append(path)

    person_embeddings = {}

    print("Building references...\n")

    for user_id, files in references.items():
        embeddings = []

        for file_path in files:
            emb = get_embedding(model, file_path, device)
            embeddings.append(emb)

        mean_embedding = np.mean(embeddings, axis=0)
        person_embeddings[user_id] = mean_embedding

        print(
            f"User {user_id}: "
            f"{len(files)} reference files"
        )

    return person_embeddings


# =========================
# PREDICT
# =========================
def predict(model, device):
    references = build_references(model, device)

    test_path = os.path.join(TEST_FOLDER, TEST_FILE)

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find test file: {test_path}"
        )

    print(f"\nTesting file: {test_path}\n")

    test_embedding = get_embedding(
        model,
        test_path,
        device
    )

    best_user = None
    best_distance = float("inf")

    print("Distances:")

    for user_id, ref_embedding in references.items():
        d = distance(test_embedding, ref_embedding)

        print(f"User {user_id}: {d:.4f}")

        if d < best_distance:
            best_distance = d
            best_user = user_id

    print(f"\nPrediction: User {best_user}")


# =========================
# MAIN
# =========================
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MovementGRU().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    predict(model, device)


if __name__ == "__main__":
    main()