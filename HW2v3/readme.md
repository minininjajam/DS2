
Название: Siamese Autoencoder for Face Verification (LFW Dataset).

Описание: Реализация сиамского автоэнкодера «с нуля» на PyTorch для задачи верификации лиц.

Ключевые фишки:
Собственная архитектура Encoder-Decoder (без предобученных моделей).
Комбинированная функция потерь: MSE + Contrastive Loss.
Эксперимент на «уверенность» модели с использованием фото Арнольда Шварценеггера и его двойника.
Результат: Точность ~72% на 10 эпохах.

https://colab.research.google.com/drive/1YZkEYSg7xmjBiHRt8MOdoHqGli-aKk9T#scrollTo=5Mg76BRI7kcN
