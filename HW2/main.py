import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt

# 1. Настройка системы
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Статус системы: Использование {device}")

model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

preprocess = T.Compose([
    T.Resize(160),
    T.CenterCrop(160),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_emb(path):
    try:
        img = Image.open(path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(img_t).flatten()
    except:
        return None

# Пути к данным
my_photo = "me.jpg"
lfw_dir = "lfw-deepfunneled"
cache_file = "facenet_embeddings.pth"

# 2. Кэширование данных
if os.path.exists(cache_file):
    print("Статус: Загрузка векторов из кэша...")
    star_data = torch.load(cache_file)
else:
    print("Статус: Сканирование датасета LFW (первый запуск)...")
    star_data = []
    count = 0
    for root, _, files in os.walk(lfw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                emb = get_emb(path)
                if emb is not None:
                    star_data.append({
                        'name': os.path.basename(root).replace('_', ' '),
                        'vec': emb.cpu(),
                        'path': path
                    })
                count += 1
                if count % 1000 == 0:
                    print(f"Обработано изображений: {count}...")
    torch.save(star_data, cache_file)
    print("Статус: Кэш успешно создан.")

# 3. Поиск ТОП-3 совпадений
if not os.path.exists(my_photo):
    print(f"Ошибка: Файл {my_photo} не найден.")
    exit()

my_v = get_emb(my_photo)
if my_v is None:
    print("Ошибка: Не удалось обработать ваше фото.")
    exit()
my_v = my_v.cpu()

results = []
for item in star_data:
    dist = torch.dist(my_v, item['vec']).item()
    results.append({'name': item['name'], 'dist': dist, 'path': item['path']})

results.sort(key=lambda x: x['dist'])
top_3 = results[:3]

# 4. Вывод данных и визуализация
print("\n--- Результаты поиска (Топ-3) ---")
with open("results.txt", "w", encoding="utf-8") as f:
    for i, res in enumerate(top_3):
        # Рассчитываем процент для наглядности (на базе FaceNet)
        score = max(0, 100 - (res['dist'] * 45))

        # Строка для вывода
        output = f"{i + 1}. {res['name']} | Сходство: {score:.1f}% | Евклид: {res['dist']:.4f}"
        print(output)
        f.write(output + "\n")

# Создаем полотно: 1е наше фото + 3 двойника
fig, axes = plt.subplots(1, 4, figsize=(20, 8))

# 1. Наше фото
img_me = Image.open(my_photo)
axes[0].imshow(img_me)
axes[0].set_title("ВАШЕ ФОТО", fontsize=12, fontweight='bold', pad=15)
axes[0].axis('off')

# 2. Отрисовка Топ-3 звезд
for i, res in enumerate(top_3):
    score = max(0, 100 - (res['dist'] * 45))
    img_star = Image.open(res['path'])

    
    axes[i + 1].imshow(img_star)

    # Формируем заголовок для каждой звезды с Евклидовым расстоянием
    axes[i + 1].set_title(
        f"№{i + 1}: {res['name']}\n"
        f"Сходство: {score:.1f}%\n"
        f"Евклид: {res['dist']:.4f}",
        fontsize=11,
        color='darkblue',
        pad=10
    )
    axes[i + 1].axis('off')

# Общий заголовок всего графика
plt.suptitle("Результаты распознавания лиц: Сравнение с базой LFW", fontsize=16, y=0.98, fontweight='bold')

# Подстройка отступов=
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Сохранение финального отчета
plt.savefig("comparison_top3_report.png", dpi=300)

print("\nГрафик успешно сформирован и сохранен как 'comparison_top3_report.png'")
print("Текстовый отчет сохранен в 'results.txt'")

# Показ окна
plt.show()
