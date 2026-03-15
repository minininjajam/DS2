import json
import torch
import torch.nn as nn
import numpy as np
import logging
import nltk
from nltk.stem.snowball import RussianStemmer
from model import NeuralNet

# Настройка логирования обучения
logging.basicConfig(level=logging.INFO, filename="train_log.txt", filemode="w",
                    format="%(asctime)s - %(message)s")

stemmer = RussianStemmer()
nltk.download('punkt')
nltk.download('punkt_tab')



def tokenize(s): return nltk.word_tokenize(s)


def stem(w): return stemmer.stem(w.lower())


def bag_of_words(s_tokens, words):
    s_words = [stem(w) for w in s_tokens]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in s_words: bag[idx] = 1
    return bag


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set([stem(w) for w in all_words if w not in ['?', '!', '.']]))
tags = sorted(set(tags))

X_train, y_train = [], []
for (pattern, tag) in xy:
    X_train.append(bag_of_words(pattern, all_words))
    y_train.append(tags.index(tag))

X_train, y_train = torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)).to(torch.long)

# Обучение
model = NeuralNet(len(all_words), 8, len(tags))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        # Считаем МЕТРИКУ Accuracy (Точность)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_train).sum().item() / y_train.size(0)
        msg = f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}'
        print(msg)
        logging.info(msg)

torch.save({"model_state": model.state_dict(), "input_size": len(all_words),
            "hidden_size": 8, "output_size": len(tags),
            "all_words": all_words, "tags": tags}, "data.pth")
