import os
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

# Импорт твоих модулей (должны быть в папке)
from model import NeuralNet
from train import bag_of_words, tokenize

# --- 1. ЛОГИРОВАНИЕ ---
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
TOKEN = "8637724226:AAEpRmDEYlgwVXnzqrFkNQDuM50b-omtn_s"  # Или os.getenv("BOT_TOKEN")
ADMIN_ID = 0
FILE_PATH = "orders.csv"
QUESTIONS_FILE = "questions.csv"

# Данные каталога
CATALOG = {
    "1": {"name": "🍯 Мёд",
          "items": [{"name": "Гречишный (0.5л)", "price": 25}, {"name": "Цветочный (0.5л)", "price": 20},
                    {"name": "Лесной (0.5л)", "price": 20}]},
    "2": {"name": "🧼 Мыло", "items": [{"name": "Медовое", "price": 6}, {"name": "Лаванда и прополис", "price": 8},
                                      {"name": "Овсяное скраб", "price": 7}]},
    "3": {"name": "🕯 Свечи", "items": [{"name": "Из вощины", "price": 5}, {"name": "Фигурная 'Улей'", "price": 12},
                                       {"name": "Набор для декора", "price": 25}]}
}
CATEGORY_MAP = {"1": "1", "мед": "1", "мёд": "1", "медок": "1", "2": "2", "мыло": "2", "мыльце": "2", "3": "3",
                "свечи": "3", "свеча": "3"}

# --- 3. ЗАГРУЗКА ИИ ---
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


# --- 6. ФУНКЦИИ ---
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


bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())


# --- 7. ОБРАБОТЧИКИ ---

@dp.message(Command("start"))
@dp.message(F.text == "🏠 В главное меню")
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    resps = [i['responses'] for i in intents_data['intents'] if i['tag'] == 'greeting']
    await message.answer(random.choice(random.choice(resps)), reply_markup=get_main_menu())


@dp.message(F.text == "🚚 Доставка")
async def delivery(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    last = u_data.get("last_d", "")
    jokes = [
        "🚛 Доставка Белпочтой, Европочтой. Летим к вам быстрее роя! 🐝💨",
        "🚛 Пакуем мёд так надёжно, что он выдержит падение с цветка! Доставка Белпочтой и Европочтой. 🍯📦",
        "🚛 Доставка по стране. Мы бы отправили пчёл, но они отвлекаются на цветы! 📦🌸"
    ]
    current = random.choice([j for j in jokes if j != last])
    await state.update_data(last_d=current)
    await message.answer(current, reply_markup=get_main_menu())


@dp.message(F.text == "🐝 О пасеке")
async def about(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    last = u_data.get("last_a", "")
    jokes = [
        "🏠 Наша пасека — это закрытый клуб для пчел-трудоголиков! 🐝",
        "Мы семейная команда. Пчёлки поют мёду колыбельные! ✨",
        "У нас всё натуральное — даже укусы пасечника сертифицированы! ❤️",
        "Наши пчелы настолько суровы, что цветы сами отдают им нектар! 🍯"
    ]
    current = random.choice([j for j in jokes if j != last])
    await state.update_data(last_a=current)
    await message.answer(current, reply_markup=get_main_menu())


# --- ВОПРОСЫ / ОПТ ---
@dp.message(F.text == "✍️ Задать вопрос")
async def ask_question(message: types.Message, state: FSMContext):
    await state.set_state(QuestionProcess.waiting_for_text)
    await message.answer("📝 Если не нашли товар или нужен **ОПТ** — напишите вопрос ниже 👇", parse_mode="Markdown",
                         reply_markup=ReplyKeyboardRemove())


@dp.message(QuestionProcess.waiting_for_text)
async def get_q_text(message: types.Message, state: FSMContext):
    await state.update_data(q_text=message.text)
    await message.answer("📞 Напишите ваш телефон для связи:")
    await state.set_state(QuestionProcess.waiting_for_phone)


@dp.message(QuestionProcess.waiting_for_phone)
async def get_q_phone(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    save_to_csv(QUESTIONS_FILE,
                {"Дата": datetime.now().strftime("%d.%m.%Y %H:%M"), "Вопрос": u_data['q_text'], "Тел": message.text})
    logger.info(f"ВОПРОС от @{message.from_user.username}: {u_data['q_text']}")
    await message.answer("✅ Отправлено! Пасечник свяжется с вами.", reply_markup=get_main_menu())
    await state.clear()


# --- КОРЗИНА И КАТАЛОГ ---
@dp.message(F.text.in_(["🍯 Наш ассортимент", "🛍 Добавить еще товар"]))
async def start_catalog(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    if "cart" not in u_data:
        await state.update_data(cart=[], total_price=0)
    await message.answer("🛒 **Выберите категорию (цифра или название):**\n1. Мёд\n2. Мыло\n3. Свечи",
                         parse_mode="Markdown", reply_markup=get_main_menu())
    await state.set_state(OrderProcess.choosing_category)


@dp.message(OrderProcess.choosing_category)
async def select_category(message: types.Message, state: FSMContext):
    u_input = message.text.lower().strip()
    # Умная проверка опечатки в категории
    match, score = process.extractOne(u_input, CATEGORY_MAP.keys())
    cat_id = CATEGORY_MAP[match] if score > 80 else None

    if not cat_id:
        await message.answer("❌ Введите номер (1, 2, 3) или название (Мёд, Мыло, Свечи):")
        return

    category = CATALOG[cat_id]
    await state.update_data(current_cat=cat_id)
    items_text = f"✨ **{category['name']}**\nВведите номер:\n" + "\n".join(
        [f"{idx + 1}. {i['name']} - {i['price']}р" for idx, i in enumerate(category['items'])]) + "\n\n4. 🔙 Назад"
    await message.answer(items_text, parse_mode="Markdown")
    await state.set_state(OrderProcess.choosing_item)


@dp.message(OrderProcess.choosing_item)
async def select_item(message: types.Message, state: FSMContext):
    choice = message.text.strip().lower()
    if choice in ["4", "назад", "каталог"]:
        await start_catalog(message, state)
        return
    u_data = await state.get_data()
    items = CATALOG[u_data['current_cat']]['items']

    # Умная проверка опечатки в товаре
    sel_p = None
    if choice.isdigit() and 0 < int(choice) <= len(items):
        sel_p = items[int(choice) - 1]
    else:
        match, score = process.extractOne(choice, [i['name'] for i in items])
        if score > 70:
            sel_p = next(i for i in items if i['name'] == match)

    if not sel_p:
        await message.answer("❌ Не найдено. Введите номер или название.")
        return

    cart = u_data.get("cart", [])
    cart.append(sel_p['name'])
    total = u_data.get("total_price", 0) + sel_p['price']
    await state.update_data(cart=cart, total_price=total)
    await message.answer(f"📦 Добавлено! Итого в корзине: **{total} р.**", reply_markup=get_cart_menu(),
                         parse_mode="Markdown")
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
    save_to_csv(FILE_PATH, {"Дата": datetime.now(), "Товары": items_str, "Сумма": total, "Тел": message.text})
    logger.info(f"ЗАКАЗ: @{message.from_user.username} купил {items_str} на {total}р")
    await message.answer(f"🎉 Заказ на сумму **{total} р.** принят!", reply_markup=get_main_menu(),
                         parse_mode="Markdown")
    await state.clear()


# --- 8. УМНЫЙ ЧАТ (ИИ) ---
@dp.message()
async def smart_chat(message: types.Message, state: FSMContext):
    if not message.text: return
    # Исправляем опечатки перед подачей в нейросеть
    tokens = fix_typos(message.text.lower(), all_words)
    X = torch.from_numpy(bag_of_words(tokens, all_words)).to(device)
    output = model(X.unsqueeze(0))
    prob, pred = torch.max(torch.softmax(output, dim=1), dim=1)
    tag = tags[pred.item()]

    if prob.item() > 0.70:
        intent = [i for i in intents_data['intents'] if i['tag'] == tag][0]
        # Защита от повтора ответа
        u_data = await state.get_data()
        last_r = u_data.get(f"last_r_{tag}", "")
        possible = [r for r in intent['responses'] if r != last_r]
        ans = random.choice(possible)
        await state.update_data({f"last_r_{tag}": ans})

        logger.info(f"ИИ: {message.text} -> {tag}")
        await message.answer(ans, reply_markup=get_main_menu())
        if tag in ["honey_info", "soap_info", "candles_info", "catalog"]:
            await start_catalog(message, state)
    else:
        logger.warning(f"НЕ ПОНЯЛ: {message.text}")
        await message.answer("Хмм, интересно... 🤔 Напишите пасечнику через кнопку 'Задать вопрос'!",
                             reply_markup=get_main_menu())


async def main():
    print("Бот готов и ведет логи! 🐝")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
