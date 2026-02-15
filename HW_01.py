import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Загрузка данных
print("Загрузка базы данных...")
df = pd.read_csv('GlobalLandTemperaturesByCountry.csv')

# --- ИНТЕРАКТИВНЫЙ ВВОД СТРАНЫ ---
all_countries = df['Country'].unique() # Список всех доступных стран
target_country = input("Введите название страны на английском (например, Belarus, Poland, Russia): ").strip()

if target_country not in all_countries:
    print(f"Ошибка: Страна '{target_country}' не найдена в базе.")
    print("Примеры доступных стран:", ", ".join(all_countries[:10])) # Показываем первые 10 для примера
    exit()

# 2. Фильтрация и очистка
df_filtered = df[df['Country'] == target_country].copy()
df_filtered = df_filtered.dropna(subset=['AverageTemperature'])

# Преобразование дат
df_filtered['dt'] = pd.to_datetime(df_filtered['dt'])
df_filtered['year'], df_filtered['month'] = df_filtered['dt'].dt.year, df_filtered['dt'].dt.month

# 3. Разделение и обучение (80/20)
X = df_filtered[['year', 'month']]
y = df_filtered['AverageTemperature']
split = int(len(df_filtered) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Обучение модели для страны: {target_country}...")
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# 4. Расчет метрик
mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
print(f"Точность модели (R2): {r2:.4f}")
print(f"Средняя ошибка (MAE): {mae:.2f} °C")

# --- ИНТЕРАКТИВНЫЙ ВВОД ГОДА ---
try:
    target_year = int(input(f"Введите год для прогноза в {target_country}: "))
except ValueError:
    print("Ошибка: введите целое число.")
    exit()

# 5. Прогноз
future = pd.DataFrame({'year': target_year, 'month': range(1, 13)})
future['Temp'] = model.predict(future)

print(f"\nПрогноз для {target_country} на {target_year} год:")
print(future[['month', 'Temp']].to_string(index=False))

# 6. Визуализация
plt.figure(figsize=(10, 5))
plt.plot(future['month'], future['Temp'], 'o-g', label='Прогноз')
plt.title(f'Прогноз температуры: {target_country} ({target_year})')
plt.xlabel('Месяц')
plt.ylabel('Температура (°C)')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.axhline(0, color='red', alpha=0.3)
plt.legend()
plt.show()