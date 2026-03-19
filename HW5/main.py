import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# --- КОНФИГУРАЦИЯ ПРОЕКТА ---
PATH_TO_DATA = './dataset/landscapes'  # Директория с обучающей выборкой
CHECKPOINT_PATH = 'pix2pix_landscape.pth' # Файл сохранения весов модели
BATCH_SIZE = 1  # Оптимальный размер батча для архитектуры Pix2Pix
EPOCHS = 100    # Общее количество эпох обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. ПОДГОТОВКА ДАННЫХ ---
class LandscapeDataset(Dataset):
    """
    Класс для загрузки данных и автоматической генерации пар 'Набросок-Фото'.
    Входные изображения преобразуются в контурные рисунки с помощью детектора Canny.
    """
    def __init__(self, root_dir):
        # Фильтрация только графических файлов в указанной директории
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.root_dir = root_dir
        # Нормализация данных в диапазон [-1, 1] для корректной работы Tanh в генераторе
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, idx):
        # Загрузка и предварительная обработка целевого изображения
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Алгоритм генерации искусственного скетча (входные данные для модели)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150) # Выделение границ объектов
        edges = cv2.bitwise_not(edges)      # Инверсия для получения черных линий на белом фоне
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return self.transform(Image.fromarray(edges)), self.transform(Image.fromarray(img))


# --- 2. АРХИТЕКТУРА МОДЕЛИ (U-Net) ---
class UNetBlock(nn.Module):
    """Универсальный блок свертки/деконволюции с BatchNorm и активацией."""
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False) if down
            else nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2) if down else nn.ReLU()
        )

    def forward(self, x): 
        return self.conv(x)


class Generator(nn.Module):
    """
    Генератор на базе U-Net. Использует Skip-connections для передачи 
    высокочастотных деталей с входного изображения напрямую в декодер.
    """
    def __init__(self):
        super().__init__()
        # Энкодер (сжатие признаков)
        self.d1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        
        # Декодер (восстановление изображения)
        self.u1 = UNetBlock(256, 128, down=False)
        self.u2 = UNetBlock(256, 64, down=False) # Вход 256 каналов из-за конкатенации слоев
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1), 
            nn.Tanh() # Ограничение выходных значений диапазоном [-1, 1]
        )

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.u1(x3)
        # Реализация Skip-connections через torch.cat
        x5 = self.u2(torch.cat([x4, x2], 1))
        return self.final(torch.cat([x5, x1], 1))


class Discriminator(nn.Module):
    """
    Дискриминатор PatchGAN. Оценивает реалистичность локальных патчей изображения.
    Принимает на вход конкатенацию скетча и целевого фото.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1) # Выходная карта вероятностей
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))


# --- 3. ПРОЦЕСС ОБУЧЕНИЯ ---
def train():
    dataset = LandscapeDataset(PATH_TO_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # Использование оптимизатора Adam с параметрами, рекомендованными для Pix2Pix
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss() # Least Squares Loss для стабильности GAN
    criterion_l1 = nn.L1Loss()   # Попиксельная разница для сохранения структуры

    # Загрузка чекпоинта при его наличии
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH)
        gen.load_state_dict(ckpt['gen_state'])
        disc.load_state_dict(ckpt['disc_state'])
        start_epoch = ckpt['epoch']

    for epoch in range(start_epoch, EPOCHS):
        for i, (sketch, real) in enumerate(loader):
            sketch, real = sketch.to(DEVICE), real.to(DEVICE)

            # Обновление параметров Дискриминатора
            fake = gen(sketch)
            d_real = disc(sketch, real)
            d_fake = disc(sketch, fake.detach())

            loss_d = (criterion_gan(d_real, torch.ones_like(d_real)) +
                      criterion_gan(d_fake, torch.zeros_like(d_fake))) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Обновление параметров Генератора
            d_fake_g = disc(sketch, fake)
            loss_g_gan = criterion_gan(d_fake_g, torch.ones_like(d_fake_g))
            # Взвешенный коэффициент L1 Loss для минимизации артефактов
            loss_g_l1 = criterion_l1(fake, real) * 100 

            loss_g = loss_g_gan + loss_g_l1

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss D: {loss_d.item():.4f} Loss G: {loss_g.item():.4f}")

        # Сохранение весов и состояния оптимизаторов после каждой эпохи
        torch.save({
            'epoch': epoch + 1,
            'gen_state': gen.state_dict(),
            'disc_state': disc.state_dict()
        }, CHECKPOINT_PATH)


def test_my_drawing(image_path):
    """Инференс модели на произвольном входном изображении."""
    gen = Generator().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH)
    gen.load_state_dict(ckpt['gen_state'])
    gen.eval()

    # Загрузка и нормализация тестового скетча
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    sketch = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake = gen(sketch)
        # Денормализация тензора для сохранения в файл
        fake = (fake.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
        fake = (fake * 255).astype(np.uint8)
        Image.fromarray(fake).save('result_landscape.jpg')


if __name__ == "__main__":
    # train() # Запуск цикла обучения
    test_my_drawing('my_sketch.jpg') # Запуск тестирования на одном файле
