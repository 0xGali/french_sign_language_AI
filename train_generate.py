import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Config
# -----------------------------
MAX_FRAMES = 50
FEATURES_PER_FRAME = 84
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset
# -----------------------------
class SignLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# -----------------------------
# Preprocessing
# -----------------------------
def pad_sequence(seq, max_len=MAX_FRAMES):
    if len(seq) > max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - len(seq), seq.shape[1]))
    return np.vstack([seq, pad])

def tokenize_sequence(sequence, max_points=21):
    frames = []
    for frame in sequence:
        features = []
        if not isinstance(frame, list) or len(frame) < 2:
            frame = [[], []]  # si frame mal formée
        for hand in frame[:2]:  # seulement 2 mains
            hand_points = []
            for point in hand:
                if isinstance(point, (list, tuple)) and len(point) >= 3:
                    hand_points.append(point)
            # pad pour avoir exactement max_points
            while len(hand_points) < max_points:
                hand_points.append([0,0.0,0.0])
            hand_points = hand_points[:max_points]
            for _, x, y in hand_points:
                features.extend([x, y])
        frames.append(features)
    return np.array(frames)

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X, y = [], []
    for word, sequences in data.items():
        for seq in sequences:
            X.append(pad_sequence(tokenize_sequence(seq)))
            y.append(word)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = np.array(X)
    y = np.array(y_encoded)
    print(f"Dataset size: {len(X)}")
    print(f"Classes: {list(le.classes_)}")
    return X, y, le

# -----------------------------
# Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # prendre la dernière sortie
        out = self.fc(out)
        return out

# -----------------------------
# Training
# -----------------------------
def train_model(json_path):
    X, y, le = load_data(json_path)
    dataset = SignLanguageDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMClassifier(input_size=FEATURES_PER_FRAME,
                           hidden_size=HIDDEN_SIZE,
                           num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    return model, le

# -----------------------------
# Prediction
# -----------------------------
def predict(model, le, sequence):
    model.eval()
    with torch.no_grad():
        seq_padded = pad_sequence(tokenize_sequence(sequence))
        input_tensor = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        outputs = model(input_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        return le.inverse_transform([pred_idx])[0]

# -----------------------------
# Exemple d'utilisation
# -----------------------------
if __name__ == "__main__":
    model, le = train_model("coordonnees_mots.json")

    # Exemple de prédiction
    sample_sequence = [[[ [0,0.5,0.5] for _ in range(21)], [ [0,0.5,0.5] for _ in range(21)] ]]  # shape (1 frame, 2 hands, 21 points)
    word_pred = predict(model, le, sample_sequence)
    print("Mot prédit:", word_pred)
    torch.save(model.state_dict(), "sign_language_lstm.pth")
    torch.save(le, "label_encoder.pkl")