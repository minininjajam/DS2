import asyncio
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import urllib.parse
import html
import random

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

# =========================================================
# 1. НАСТРОЙКИ
# =========================================================

API_TOKEN = '8646744882:AAGgyHAnuDNcfU0YMp5tnUFJJ1UIkufE0_k'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# =========================================================
# 2. ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ
# =========================================================

print(" Загрузка моделей...")

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

df_vibe = pd.read_csv('processed_music.csv').reset_index(drop=True)
embeddings = np.load('music_embeddings.npy').astype('float32')
df_full = pd.read_csv('dataset.csv').dropna(subset=['track_name', 'artists'])

# =========================================================
# 3. LSTM МОДЕЛЬ
# =========================================================

checkpoint = torch.load('my_lstm_model.pth', map_location='cpu')

genres_list = checkpoint['genres']
vocab = checkpoint['vocab']

class MyMusicLSTM(nn.Module):
    def __init__(self, v_size, n_classes):
        super().__init__()
        self.emb = nn.Embedding(v_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

lstm = MyMusicLSTM(checkpoint['vocab_size'], checkpoint['num_classes'])
lstm.load_state_dict(checkpoint['model_state'])
lstm.eval()

# =========================================================
# 4. ПАМЯТЬ И КЭШ
# =========================================================

cache = {}
user_history = {}
user_offset = {}

# =========================================================
# 5. СЛОВАРИ
# =========================================================

emoji_map = {
    "😢": "sad emotional piano",
    "😭": "very sad slow",
    "🔥": "energetic hype gym",
    "💪": "workout aggressive",
    "🌙": "night calm ambient",
    "😎": "cool chill vibe",
    "❤️": "romantic love song",
    "😴": "sleep calm ambient",
}

greetings = ["привет", "хай", "hello", "hi", "здравствуй"]
farewells = ["пока", "bye", "goodbye", "увидимся", "до свидания"]

moods = {
    "sad": "sad emotional piano",
    "hype": "energetic gym aggressive",
    "chill": "lofi chill relax study",
    "night": "night dark ambient"
}

# =========================================================
# 6. РЕКОМЕНДАЦИИ
# =========================================================

def get_recommendations(query_text, offset=0):
    if query_text in cache:
        v = cache[query_text]
    else:
        v = embedder.encode([query_text]).astype('float32')
        cache[query_text] = v

    distances = cdist(v, embeddings, metric='cosine')[0]
    sorted_idx = np.argsort(distances)

    return df_vibe.loc[sorted_idx[offset:offset+5]]

# =========================================================
# 7. YOUTUBE
# =========================================================

def make_youtube(track, artist):
    query = urllib.parse.quote(f"{artist} {track} official audio")
    return f"https://www.youtube.com/results?search_query={query}"

# =========================================================
# 8. LSTM ЖАНР
# =========================================================

def predict_genre(name):
    ids = [vocab.get(w, 0) for w in name.lower().split()]
    ids = (ids + [0]*10)[:10]

    with torch.no_grad():
        g_idx = torch.argmax(lstm(torch.tensor([ids])), dim=1).item()

    return genres_list[g_idx]

# =========================================================
# 9. ФОРМИРОВАНИЕ ОТВЕТА
# =========================================================

def build_response(tracks):
    styles = [
        "Поймал твой вайб… слушай:",
        "Кажется, тебе это зайдет:",
        "Вот что сейчас идеально подходит:",
        "Попробуй это:"
    ]

    res = f"<b>{random.choice(styles)}</b>\n\n"
    buttons = []

    for _, row in tracks.iterrows():
        name = html.escape(str(row['track_name']))
        art = html.escape(str(row['artists']))

        genre = predict_genre(name)
        yt = make_youtube(name, art)

        res += f"🎵 <b>{name}</b>\n"
        res += f"👤 {art}\n"
        res += f"🎹 <code>{genre}</code>\n\n"

        buttons.append([
            InlineKeyboardButton(
                text=f"️ {name[:20]}",
                url=yt
            )
        ])

    buttons.append([
        InlineKeyboardButton(text=" Еще", callback_data="more")
    ])

    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    return res, kb

# =========================================================
# 10. START
# =========================================================

@dp.message(Command("start"))
async def start(m: types.Message):
    import os
    from aiogram.types import FSInputFile

    # Текст описания возможностей системы (без смайликов и упоминания языков)
    description_text = (
        "<b>VibeSynth AI: Интеллектуальная система музыкальных рекомендаций</b>\n\n"
        "Эта нейросеть анализирует контекст вашего запроса и находит музыку, "
        "соответствующую заданному состоянию или атмосфере.\n\n"
        "<b>Как это работает:</b>\n"
        "Опишите словами, что вы чувствуете, или укажите желаемый стиль. "
        "Алгоритм сопоставит ваш запрос с базой данных и предложит наиболее "
        "подходящие композиции.\n\n"
        "<b>Доступные команды для быстрого поиска:</b>\n"
        "/sad /hype /chill /night"
    )

    # Проверяем файл и отправляем
    if os.path.exists("welcome.gif"):
        animation = FSInputFile("welcome.gif")
        await m.answer_animation(
            animation=animation,
            caption=description_text,
            parse_mode="HTML"
        )
    else:
        # Если файл не найден
        await m.answer(description_text, parse_mode="HTML")


# =========================================================
# 11. ОБРАБОТКА СООБЩЕНИЙ
# =========================================================

@dp.message()
async def handle(m: types.Message):
    if not m.text:
        return

    text_raw = m.text.lower().strip()

    # приветствие
    if text_raw in greetings:
        await m.answer("👋 Привет! Давай найдем тебе идеальный трек")
        return

    # прощание
    if text_raw in farewells:
        await m.answer("👋 Пока! Если захочешь музыку — я всегда тут ")
        return

    status = await m.answer("Подбираю что-то интересное...")

    # эмодзи
    if text_raw in emoji_map:
        text = emoji_map[text_raw]
    else:
        text = text_raw

    # команды
    if text.startswith("/"):
        cmd = text.replace("/", "")
        if cmd in moods:
            text = moods[cmd]

    user_history.setdefault(m.from_user.id, []).append(text)
    user_offset[m.from_user.id] = 0

    try:
        tracks = get_recommendations(text, offset=0)
        res, kb = build_response(tracks)

        await status.delete()
        await m.answer(res, parse_mode="HTML", reply_markup=kb)

    except Exception as e:
        print("Ошибка:", e)
        await m.answer("Попробуй описать настроение чуть по-другому")

# =========================================================
# 12. КНОПКА "ЕЩЕ"
# =========================================================

@dp.callback_query(lambda c: c.data == "more")
async def more(callback: types.CallbackQuery):
    await callback.answer("Сейчас найду еще ")

    user_id = callback.from_user.id
    last_query = user_history.get(user_id, ["music"])[-1]

    user_offset[user_id] = user_offset.get(user_id, 0) + 5

    tracks = get_recommendations(last_query, offset=user_offset[user_id])

    if tracks.empty:
        user_offset[user_id] = 0
        tracks = get_recommendations(last_query, offset=0)

    res, kb = build_response(tracks)

    res = "Хочешь что-то еще в этом духе?\n\n" + res

    await callback.message.answer(res, parse_mode="HTML", reply_markup=kb)

# =========================================================
# 13. ЗАПУСК
# =========================================================

if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))