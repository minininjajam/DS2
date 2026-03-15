import torch
import random
import json
import pandas as pd
import asyncio
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.filters import Command
from fuzzywuzzy import process

# Импорт архитектуры и функций обработки из твоих файлов (по примеру Лёбера)
from model import NeuralNet
from train import bag_of_words, tokenize

# --- 1. ЛОГИРОВАНИЕ (Требование задания) ---
# Логирование вопросов/ответов и работы бота
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. НАСТРОЙКИ ---
TOKEN = "8637724226:AAEpRmDEYlgwVXnzqrFkNQDuM50b-omtn_s"
ADMIN_ID = 0  # Твой ID для уведомлений
FILE_PATH = "orders.csv"
QUESTIONS_FILE = "questions.csv"

# Данные каталога с ценами
CATALOG = {
    "1": {"name": "🍯 Мёд", "items": [{"name": "Гречишный (0.5л)", "price": 25}, {"name": "Цветочный (0.5л)", "price": 20}, {"name": "Лесной (0.5л)", "price": 20}]},
    "2": {"name": "🧼 Мыло", "items": [{"name": "Мыло 'Медовое'", "price": 6}, {"name": "Мыло 'Лаванда'", "price": 8}, {"name": "Мыло 'Скраб'", "price": 7}]},
    "3": {"name": "🕯 Свечи", "items": [{"name": "Свеча из вощины", "price": 5}, {"name": "Фигурная 'Улей'", "price": 12}, {"name": "Набор для декора", "price": 25}]}
}
CATEGORY_MAP = {"1": "1", "мед": "1", "мёд": "1", "2": "2", "мыло": "2", "3": "3", "свечи": "3"}

# --- 3. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ (PyTorch) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

data = torch.load("data.pth")
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

all_words, tags = data['all_words'], data['tags']

# --- 4. СОСТОЯНИЯ (FSM) ---
class OrderProcess(StatesGroup):
    choosing_category = State()
    choosing_item = State()
    more_items_choice = State()
    waiting_for_phone = State()

class QuestionProcess(StatesGroup):
    waiting_for_text = State()
    waiting_for_phone = State()

# --- 5. КЛАВИАТУРЫ ---
def get_main_menu():
    kb = [[KeyboardButton(text="🍯 Наш ассортимент"), KeyboardButton(text="🚚 Доставка")],
          [KeyboardButton(text="🐝 О пасеке"), KeyboardButton(text="✍️ Задать вопрос")]]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_cart_menu():
    kb = [[KeyboardButton(text="🛍 Добавить еще товар")], [KeyboardButton(text="✅ Оформить всё")]]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

# --- 6. ФУНКЦИИ СОХРАНЕНИЯ ---
def save_to_csv(file, data_dict):
    try:
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([data_dict])
    df.to_csv(file, index=False, encoding='utf-8-sig')

def fix_typos(text, known_words):
    words = tokenize(text)
    fixed = []
    for w in words:
        match, score = process.extractOne(w, known_words)
        fixed.append(match if score > 80 else w)
    return fixed

# --- 7. ИНИЦИАЛИЗАЦИЯ И ОБРАБОТЧИКИ ---
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())

@dp.message(Command("start"))
@dp.message(F.text == "🏠 В главное меню")
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    responses = [i['responses'] for i in intents_data['intents'] if i['tag'] == 'greeting']
    await message.answer(random.choice(random.choice(responses)), reply_markup=get_main_menu())

@dp.message(F.text == "🚚 Доставка")
async def delivery(message: types.Message, state: FSMContext):
    await state.clear()
    jokes = ["🚛 Доставка Белпочтой, Европочтой. Летим к вам быстрее роя! 🐝💨", "🚛 Доставка Белпочтой/Европочтой. Пакуем надёжно! 🍯📦"]
    await message.answer(random.choice(jokes), reply_markup=get_main_menu())

@dp.message(F.text == "🐝 О пасеке")
async def about(message: types.Message, state: FSMContext):
    await state.clear()
    jokes = ["🏠 Наша пасека — место пчел-трудоголиков! 🐝", "Мы семейная команда, пчелы поют меду колыбельные! ✨"]
    await message.answer(random.choice(jokes), reply_markup=get_main_menu())

# --- ВОПРОСЫ / ОПТ ---
@dp.message(F.text == "✍️ Задать вопрос")
async def ask_question(message: types.Message, state: FSMContext):
    await state.set_state(QuestionProcess.waiting_for_text)
    await message.answer("📝 Если не нашли товар или нужен **ОПТ** — напишите вопрос ниже 👇", parse_mode="Markdown", reply_markup=ReplyKeyboardRemove())

@dp.message(QuestionProcess.waiting_for_text)
async def get_q_text(message: types.Message, state: FSMContext):
    await state.update_data(q_text=message.text)
    await message.answer("📞 Напишите ваш телефон для связи:")
    await state.set_state(QuestionProcess.waiting_for_phone)

@dp.message(QuestionProcess.waiting_for_phone)
async def get_q_phone(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    save_to_csv(QUESTIONS_FILE, {"Дата": datetime.now(), "Вопрос": u_data['q_text'], "Тел": message.text, "Юзер": message.from_user.username})
    logger.info(f"ВОПРОС: @{message.from_user.username} спросил '{u_data['q_text']}'")
    await message.answer("✅ Отправлено! Пасечник свяжется с вами.", reply_markup=get_main_menu())
    await state.clear()

# --- КОРЗИНА И ЗАКАЗ ---
@dp.message(F.text == "🍯 Наш ассортимент")
async def start_catalog(message: types.Message, state: FSMContext):
    await state.clear()
    await state.update_data(cart=[], total_price=0)
    await show_categories(message, state)

@dp.message(F.text == "🛍 Добавить еще товар")
async def add_more(message: types.Message, state: FSMContext):
    await show_categories(message, state)

async def show_categories(message: types.Message, state: FSMContext):
    await message.answer("🛒 **Выберите категорию (1-3):**\n1. Мёд\n2. Мыло\n3. Свечи", parse_mode="Markdown", reply_markup=get_main_menu())
    await state.set_state(OrderProcess.choosing_category)

@dp.message(OrderProcess.choosing_category)
async def select_category(message: types.Message, state: FSMContext):
    u_input = message.text.lower().strip()
    cat_id = CATEGORY_MAP.get(u_input) or (process.extractOne(u_input, CATEGORY_MAP.keys())[0] if process.extractOne(u_input, CATEGORY_MAP.keys())[1] > 75 else None)
    if not cat_id:
        await message.answer("❌ Введите номер (1, 2, 3) или название:")
        return
    category = CATALOG[cat_id]
    await state.update_data(current_cat=cat_id)
    items_text = f"✨ **{category['name']}**\nВведите номер:\n" + "\n".join([f"{idx+1}. {i['name']} - {i['price']}р" for idx, i in enumerate(category['items'])]) + "\n\n4. 🔙 Назад"
    await message.answer(items_text, parse_mode="Markdown")
    await state.set_state(OrderProcess.choosing_item)

@dp.message(OrderProcess.choosing_item)
async def select_item(message: types.Message, state: FSMContext):
    choice = message.text.strip().lower()
    if choice in ["4", "назад", "каталог"]:
        await show_categories(message, state)
        return
    u_data = await state.get_data()
    items = CATALOG[u_data['current_cat']]['items']
    sel_p = items[int(choice)-1] if (choice.isdigit() and 0 < int(choice) <= len(items)) else None
    if not sel_p:
        await message.answer("❌ Не найдено. Введите номер или 4.")
        return
    cart = u_data.get("cart", [])
    cart.append(sel_p['name'])
    total = u_data.get("total_price", 0) + sel_p['price']
    await state.update_data(cart=cart, total_price=total)
    await message.answer(f"📦 Добавлено! Итого: **{total} р.**", reply_markup=get_cart_menu(), parse_mode="Markdown")
    await state.set_state(OrderProcess.more_items_choice)

@dp.message(OrderProcess.more_items_choice, F.text == "✅ Оформить всё")
async def checkout(message: types.Message, state: FSMContext):
    await message.answer("📞 Напишите ваш телефон:", reply_markup=ReplyKeyboardRemove())
    await state.set_state(OrderProcess.waiting_for_phone)

@dp.message(OrderProcess.waiting_for_phone)
async def finish_order(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    items_str = ", ".join(u_data.get("cart", []))
    total = u_data.get("total_price", 0)
    save_to_csv(FILE_PATH, {"Дата": datetime.now(), "Товары": items_str, "Сумма": total, "Тел": message.text, "Юзер": message.from_user.username})
    logger.info(f"ЗАКАЗ: @{message.from_user.username} купил {items_str} на {total}р")
    await message.answer(f"🎉 Заказ на **{total} р.** принят!", reply_markup=get_main_menu(), parse_mode="Markdown")
    await state.clear()

# --- 8. УМНЫЙ ЧАТ (Логирование ответов) ---
@dp.message()
async def smart_chat(message: types.Message, state: FSMContext):
    if not message.text: return
    u_text = message.text.lower()
    if any(w in u_text for w in ["каталог", "товары"]):
        await start_catalog(message, state)
        return
    tokens = fix_typos(u_text, all_words)
    X = torch.from_numpy(bag_of_words(tokens, all_words)).to(device)
    output = model(X.unsqueeze(0))
    prob, pred = torch.max(torch.softmax(output, dim=1), dim=1)
    tag = tags[pred.item()]
    if prob.item() > 0.70:
        resps = [i['responses'] for i in intents_data['intents'] if i['tag'] == tag][0]
        answer = random.choice(resps)
        logger.info(f"ИИ: Юзер: '{message.text}' -> Бот: '{answer}' (Tag: {tag})")
        await message.answer(answer, reply_markup=get_main_menu())
        if tag in ["honey_info", "soap_info", "candles_info"]: await start_catalog(message, state)
    else:
        logger.warning(f"ИИ НЕ ПОНЯЛ: '{message.text}'")
        await message.answer("Интересно... 🤔 Напишите пасечнику!", reply_markup=get_main_menu())

async def main():
    print("Бот готов! Логи ведутся в bot_log.txt")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
