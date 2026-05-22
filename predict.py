import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CONFIG (Synkad med din bästa träning!)
# =========================
INPUT_SIZE = 69
HIDDEN_SIZE = 128
EMBED_SIZE = 128              # Ändrad till 64 (matchar din sparade modell)
MAX_SEQ_LEN = 60
UNKNOWN_THRESHOLD = 0.51
MODEL_PATH = "movement_gru_best.pth"

DATA_FOLDER = "Data/References"
TEST_FOLDER = "Test/Final"
TEST_FILE = "ID14test3.json"


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

    # -------------------------------------------------------------
    # NYTT: SAMMA RESAMPLING SOM VID TRÄNING
    # -------------------------------------------------------------
    num_frames = seq.shape[0]
    num_features = seq.shape[1]
    
    if num_frames > 1:
        current_indices = np.linspace(0, num_frames - 1, num_frames)
        target_indices = np.linspace(0, num_frames - 1, MAX_SEQ_LEN)
        
        resampled_chunk = np.zeros((MAX_SEQ_LEN, num_features))
        for f in range(num_features):
            resampled_chunk[:, f] = np.interp(target_indices, current_indices, seq[:, f])
    else:
        resampled_chunk = np.repeat(seq, MAX_SEQ_LEN, axis=0)

    # -------------------------------------------------------------
    # RUMS-CENTRERING: Tvingar predict att titta på samma rena data
    # -------------------------------------------------------------
    centered_chunk = resampled_chunk.copy()
    for t in range(len(centered_chunk)):
        if np.all(centered_chunk[t] == 0): 
            continue
        # Index 0, 1, 2 är shoulder_x, shoulder_y, shoulder_z
        base_x, base_y, base_z = centered_chunk[t, 0], centered_chunk[t, 1], centered_chunk[t, 2]
        for p in range(0, 69, 3):
            centered_chunk[t, p] -= base_x
            centered_chunk[t, p+1] -= base_y
            centered_chunk[t, p+2] -= base_z

    return centered_chunk


# =========================
# MODEL (Exakt samma arkitektur som vid träning)
# =========================
class MovementGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            INPUT_SIZE,
            HIDDEN_SIZE,
            batch_first=True,
            num_layers=2           # Ändrad till 2 (matchar din sparade modell)
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            EMBED_SIZE
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out.mean(dim=1)      # Mean pooling
        
        emb = self.fc(out)
        return F.normalize(emb, dim=1) # Kom ihåg normaliseringen till sfären!


# =========================
# EMBEDDING
# =========================
def get_embedding(model, file_path, device):
    # load_sequence returnerar nu en färdig, centrerad array med formen (60, 69)
    chunk = load_sequence(file_path)

    tensor = torch.tensor(
        chunk,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.cpu().numpy()[0]


def distance(a, b):
    return np.linalg.norm(a - b)


# =========================
# BUILD REFERENCES (Individuella filer istället för medelvärde)
# =========================
def build_references(model, device):
    file_embeddings = []

    print("Building references...\n")

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, file)

        with open(path, "r") as f:
            raw = json.load(f)

        user_id = raw["user_id"]
        
        # Hämta embedding för denna specifika fil
        emb = get_embedding(model, path, device)
        
        file_embeddings.append({
            "user_id": user_id,
            "file_name": file,
            "embedding": emb
        })

    print(f"Loaded {len(file_embeddings)} individual reference files.\n")
    return file_embeddings


# =========================
# PREDICT (Nearest Neighbor)
# =========================
def predict(model, device):
    all_refs = build_references(model, device)

    test_path = os.path.join(TEST_FOLDER, TEST_FILE)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test file: {test_path}")

    print(f"Testing file: {test_path}\n")

    test_embedding = get_embedding(model, test_path, device)

    # Räkna ut avståndet till PRECIS VARJE referensfil
    scored_files = []
    for ref in all_refs:
        d = distance(test_embedding, ref["embedding"])
        scored_files.append({
            "user_id": ref["user_id"],
            "file_name": ref["file_name"],
            "distance": d
        })

    # Sortera så att den filen med kortast avstånd hamnar först
    scored_files = sorted(scored_files, key=lambda x: x["distance"])

    print("Top 5 closest reference files:")
    for i, res in enumerate(scored_files[:5]):
        print(f" {i+1}. User {res['user_id']} (File: {res['file_name']}) - Distance: {res['distance']:.4f}")

    # Gissa på den användare som äger den absolut närmaste filen
    best_match = scored_files[0]
    
    print("-" * 40)
    if best_match["distance"] > UNKNOWN_THRESHOLD:
        print(f"Prediction: Unknown User (Closest was User {best_match['user_id']} at {best_match['distance']:.4f})")
    else:
        print(f"Prediction: User {best_match['user_id']} (Based on file {best_match['file_name']})")


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