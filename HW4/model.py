import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Входной слой
        self.l1 = nn.Linear(input_size, hidden_size)
        # Скрытый слой
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Выходной слой (количество ответов/тегов)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Функция активации (чтобы сеть могла учить сложные зависимости)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)
