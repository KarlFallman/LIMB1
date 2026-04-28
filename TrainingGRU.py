#pytorch,jupyter
#Parametrar XYZ, ID
#INPUT HIDDEN EXPORT ONNX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# =========================
# CONFIG
# =========================
INPUT_SIZE = 69        # 23 joints * 3 coords
HIDDEN_SIZE = 64
EMBED_SIZE = 32
SEQ_LEN = 20 # Frames/Modulo
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# MODEL
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
        emb = F.normalize(emb, dim=1)  # important for similarity
        return emb


# =========================
# DUMMY DATASET (replace!)
# =========================
class DummyDataset:
    def __init__(self, num_users=5):
        self.num_users = num_users

    def sample_sequence(self, user_id):
        # simulate slightly different motion per user
        base = torch.randn(SEQ_LEN, INPUT_SIZE) + user_id * 0.5
        noise = torch.randn_like(base) * 0.1
        return base + noise

    def get_triplet(self):
        user = random.randint(0, self.num_users - 1)
        other = (user + random.randint(1, self.num_users - 1)) % self.num_users

        anchor = self.sample_sequence(user)
        positive = self.sample_sequence(user)
        negative = self.sample_sequence(other)

        return anchor, positive, negative


dataset = DummyDataset()


# =========================
# BATCH SAMPLER
# =========================
def get_batch(batch_size):
    anchors, positives, negatives = [], [], []

    for _ in range(batch_size):
        a, p, n = dataset.get_triplet()
        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )


# =========================
# TRAINING SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MovementGRU().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.TripletMarginLoss(margin=1.0)


# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()

    anchors, positives, negatives = get_batch(BATCH_SIZE)
    anchors = anchors.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)

    emb_a = model(anchors)
    emb_p = model(positives)
    emb_n = model(negatives)

    loss = criterion(emb_a, emb_p, emb_n)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "movement_gru.pth")