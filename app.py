import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="Climate Predictor", layout="wide")

st.title("🌍 Анализ и прогноз температуры по странам")
st.markdown("""
Это приложение сравнивает три алгоритма машинного обучения для предсказания климатических изменений.
""")

# 1. Загрузка данных (кешируем, чтобы не скачивать постоянно)
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com'
    df = pd.read_csv(url)
    df = df.dropna(subset=['AverageTemperature'])
    df['dt'] = pd.to_datetime(df['dt'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    # Добавляем цикличные признаки
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("Параметры")
all_countries = sorted(df['Country'].unique())
target_country = st.sidebar.selectbox("Выберите страну", all_countries, index=all_countries.index("Russia") if "Russia" in all_countries else 0)
target_year = st.sidebar.slider("Год для прогноза", 2020, 2100, 2030)

# 2. Подготовка и обучение
df_filtered = df[df['Country'] == target_country]
X_basic = df_filtered[['year', 'month']]
X_cyclic = df_filtered[['year', 'month_sin', 'month_cos']]
y = df_filtered['AverageTemperature']

with st.spinner('Обучаем модели...'):
    model_rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_basic, y)
    model_lr_simple = LinearRegression().fit(X_basic, y)
    model_lr_cyclic = LinearRegression().fit(X_cyclic, y)

# 3. Создание прогноза
future = pd.DataFrame({'year': target_year, 'month': range(1, 13)})
future['month_sin'] = np.sin(2 * np.pi * future['month'] / 12)
future['month_cos'] = np.cos(2 * np.pi * future['month'] / 12)

future['Random Forest'] = model_rf.predict(future[['year', 'month']])
future['Linear (Simple)'] = model_lr_simple.predict(future[['year', 'month']])
future['Linear (Cyclic)'] = model_lr_cyclic.predict(future[['year', 'month_sin', 'month_cos']])

# --- ВИЗУАЛИЗАЦИЯ ---
st.subheader(f"Результаты прогноза для {target_country} на {target_year} год")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(future['month'], future['Random Forest'], 'o--', label='Random Forest (Без тренда)', color='green', alpha=0.6)
ax.plot(future['month'], future['Linear (Simple)'], 's-', label='Linear Simple (Без сезонов)', color='red', alpha=0.6)
ax.plot(future['month'], future['Linear (Cyclic)'], 'D-', label='Linear Cyclic (Лучшая)', color='blue', linewidth=2)

ax.set_xticks(range(1, 13))
ax.set_xlabel("Месяц")
ax.set_ylabel("Температура °C")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- ТАБЛИЦА И ВЫВОДЫ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Данные прогноза (°C):**")
    st.dataframe(future[['month', 'Random Forest', 'Linear (Simple)', 'Linear (Cyclic)']].style.highlight_max(axis=0))

with col2:
    st.write("**Анализ алгоритмов:**")
    trend_10y = model_lr_cyclic.coef_[0] * 10
    st.info(f"📈 **Температурный тренд:** {trend_10y:+.3f}°C каждые 10 лет.")
    
    st.success("""
    **Почему Cyclic Linear Regression лучше?**
    1. Она видит глобальное потепление (в отличие от Леса).
    2. Она понимает смену времен года (в отличие от простой регрессии).
    """)
