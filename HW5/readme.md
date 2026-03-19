# Pix2Pix Landscape Generator
Нейросеть (GAN) для превращения скетчей в реалистичные пейзажи.

### Как запустить:
1. Установите зависимости: `pip install -r requirements.txt`
2. Положите свои фото в `dataset/landscapes/`
3. Запустите обучение или тест через `python main.py`

### Особенности:
* Архитектура U-Net с Skip-connections.
* Дискриминатор PatchGAN.
* Автоматическая генерация скетчей из фото с помощью Canny Edge Detector.

