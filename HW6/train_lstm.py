import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os


# --- 1. КЛАСС ДЛЯ ЗАГРУЗКИ ДАННЫХ ---
class MusicDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=10):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Превращаем название трека в список чисел (ID слов)
        text = str(self.texts[idx]).lower().split()
        ids = [self.vocab.get(w, 0) for w in text]

        # ПАДДИНГ: Дополняем нулями до нужной длины (max_len)
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        return torch.tensor(ids[:self.max_len]), torch.tensor(self.labels[idx])


# --- 2. АРХИТЕКТУРА НЕЙРОСЕТИ (LSTM) ---
class MyMusicLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_lstm():
    print("Загрузка ПОЛНОГО датасета для обучения...")
    if not os.path.exists('dataset.csv'):
        print("Ошибка: Файл dataset.csv не найден!")
        return

    df = pd.read_csv('dataset.csv').dropna(subset=['track_name', 'track_genre'])

    # Создаем список жанров и маппинг
    genres = sorted(df['track_genre'].unique().tolist())
    genre_to_id = {g: i for i, g in enumerate(genres)}

    print("Создание словаря (15 000 самых частых слов)...")
    all_words = " ".join(df['track_name'].astype(str)).lower().split()
    vocab = {word: i + 1 for i, (word, _) in enumerate(Counter(all_words).most_common(15000))}
    vocab['<PAD>'] = 0

    # Подготовка загрузчика (batch_size=128 для CPU)
    dataset = MusicDataset(
        df['track_name'].values,
        [genre_to_id[g] for g in df['track_genre']],
        vocab
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Инициализация модели
    device = torch.device('cpu')  # Форсируем CPU, так как CUDA не завелась
    model = MyMusicLSTM(len(vocab), len(genres)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    print(f"Начинаю обучение на {len(df)} строках (10 эпох)...")
    model.train()

    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Эпоха {epoch + 1}/10 | Ошибка (Loss): {avg_loss:.4f}")

    # СОХРАНЕНИЕ
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'genres': genres,
        'vocab_size': len(vocab),
        'num_classes': len(genres)
    }, 'my_lstm_model.pth')

    print("\nОБУЧЕННАЯ модель сохранена как my_lstm_model.pth!")
    print("Теперь можно запускать main.py (бота).")


if __name__ == "__main__":
    train_lstm()
