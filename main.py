"""
================================================================================
МЕГА ШПОРА ПО PANDAS: от загрузки данных до продвинутой визуализации
================================================================================
Этот скрипт содержит:
1. Создание демонстрационных данных
2. Основные операции pandas (фильтрация, группировка, сводные таблицы)
3. Работа с пропусками и дубликатами
4. Merge/Join/Concat
5. Временные ряды
6. Различные типы графиков (matplotlib + seaborn)
7. Сохранение результатов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 80)
print("🚀 ЗАПУСК МЕГА ШПОРЫ ПО PANDAS")
print("=" * 80)

# ==============================================================================
# ЧАСТЬ 1: СОЗДАНИЕ ДЕМОНСТРАЦИОННЫХ ДАННЫХ
# ==============================================================================
print("\n📊 ЧАСТЬ 1: ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
print("-" * 50)

np.random.seed(42)
n_rows = 1000

# Создаем датафрейм с разными типами данных
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n_rows, freq='D'),
    'region': np.random.choice(['Север', 'Юг', 'Запад', 'Восток'], n_rows),
    'product': np.random.choice(['Ноутбук', 'Телефон', 'Планшет', 'Монитор'], n_rows),
    'sales': np.random.normal(1000, 200, n_rows).round(2),
    'quantity': np.random.poisson(5, n_rows),
    'customer_age': np.random.randint(18, 70, n_rows),
    'rating': np.random.uniform(1, 5, n_rows).round(1),
    'is_promotion': np.random.choice([True, False], n_rows, p=[0.3, 0.7])
})

# Добавляем немного пропусков для демонстрации
df.loc[10:20, 'sales'] = np.nan
df.loc[50:55, 'rating'] = np.nan
df.loc[100:105, 'customer_age'] = np.nan

print(f"✅ Создан датафрейм размером {df.shape}")
print(f"📋 Колонки: {list(df.columns)}")
print(f"📊 Типы данных:\n{df.dtypes}")
print(f"\nПервые 5 строк:")
print(df.head())

# ==============================================================================
# ЧАСТЬ 2: ОСНОВНЫЕ ОПЕРАЦИИ PANDAS
# ==============================================================================
print("\n" + "=" * 80)
print("📈 ЧАСТЬ 2: БАЗОВЫЙ АНАЛИЗ И ОПЕРАЦИИ")
print("=" * 80)

# 2.1 Информация о данных
print("\n📌 2.1 Статистика по числовым колонкам:")
print(df.describe())

print("\n📌 2.2 Информация о пропусках:")
print(f"Пропуски в каждой колонке:\n{df.isnull().sum()}")

print("\n📌 2.3 Уникальные значения:")
for col in ['region', 'product', 'is_promotion']:
    print(f"{col}: {df[col].unique()[:5]}")

# 2.2 Фильтрация данных
print("\n📌 2.4 Фильтрация (продажи > 1200 и рейтинг > 4):")
filtered = df[(df['sales'] > 1200) & (df['rating'] > 4)]
print(f"Найдено записей: {len(filtered)}")

# 2.3 Группировка и агрегация
print("\n📌 2.5 Группировка по региону и продукту:")
grouped = df.groupby(['region', 'product']).agg({
    'sales': ['mean', 'sum', 'count'],
    'quantity': 'sum',
    'rating': 'mean'
}).round(2)
print(grouped.head(10))

# 2.4 Сводные таблицы
print("\n📌 2.6 Сводная таблица (регион x продукт):")
pivot_sales = pd.pivot_table(df, 
                             values='sales', 
                             index='region', 
                             columns='product', 
                             aggfunc='mean',
                             fill_value=0)
print(pivot_sales)

# ==============================================================================
# ЧАСТЬ 3: РАБОТА С ПРОПУСКАМИ И ДУБЛИКАТАМИ
# ==============================================================================
print("\n" + "=" * 80)
print("🛠️ ЧАСТЬ 3: ОЧИСТКА ДАННЫХ")
print("=" * 80)

# 3.1 Создаем копию для очистки
df_clean = df.copy()

# 3.2 Заполнение пропусков
print("\n📌 3.1 Заполнение пропусков:")
df_clean['sales'] = df_clean['sales'].fillna(df_clean['sales'].median())
df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].mean())
df_clean['customer_age'] = df_clean['customer_age'].fillna(df_clean['customer_age'].median())

print(f"Пропусков после очистки: {df_clean.isnull().sum().sum()}")

# 3.3 Создание новых колонок
df_clean['revenue'] = df_clean['sales'] * df_clean['quantity']
df_clean['profit'] = df_clean['revenue'] * 0.3
df_clean['month'] = df_clean['date'].dt.month
df_clean['quarter'] = df_clean['date'].dt.quarter
df_clean['weekday'] = df_clean['date'].dt.day_name()

# 3.4 Категоризация возраста
df_clean['age_group'] = pd.cut(df_clean['customer_age'], 
                               bins=[0, 25, 35, 50, 100], 
                               labels=['18-25', '26-35', '36-50', '50+'])

print("\n📌 3.2 Новые колонки созданы:")
print(df_clean[['revenue', 'profit', 'month', 'age_group']].head())

# ==============================================================================
# ЧАСТЬ 4: MERGE / JOIN / CONCAT
# ==============================================================================
print("\n" + "=" * 80)
print("🔗 ЧАСТЬ 4: ОБЪЕДИНЕНИЕ ДАННЫХ")
print("=" * 80)

# Создаем справочники
products_info = pd.DataFrame({
    'product': ['Ноутбук', 'Телефон', 'Планшет', 'Монитор'],
    'category': ['Электроника', 'Мобильные', 'Электроника', 'Периферия'],
    'margin': [0.25, 0.35, 0.20, 0.30]
})

regions_info = pd.DataFrame({
    'region': ['Север', 'Юг', 'Запад', 'Восток'],
    'manager': ['Иванов', 'Петров', 'Сидоров', 'Козлов'],
    'target': [500000, 450000, 600000, 400000]
})

print("📌 4.1 Merge с products_info:")
df_merged = df_clean.merge(products_info, on='product', how='left')
print(df_merged[['product', 'category', 'margin']].drop_duplicates())

print("\n📌 4.2 Merge с regions_info:")
df_final = df_merged.merge(regions_info, on='region', how='left')
print(df_final[['region', 'manager', 'target']].drop_duplicates())

print("\n📌 4.3 Пример Concat (добавление строк):")
new_data = pd.DataFrame({
    'date': [pd.Timestamp('2024-12-31')],
    'region': ['Центр'],
    'product': ['Ноутбук'],
    'sales': [5000],
    'quantity': [10]
})
df_concat = pd.concat([df_clean, new_data], ignore_index=True)
print(f"Размер после concat: {df_concat.shape} (было {df_clean.shape})")

# ==============================================================================
# ЧАСТЬ 5: ВРЕМЕННЫЕ РЯДЫ
# ==============================================================================
print("\n" + "=" * 80)
print("⏰ ЧАСТЬ 5: АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
print("=" * 80)

# Устанавливаем индекс по дате
df_ts = df_clean.set_index('date')
daily_sales = df_ts['sales'].resample('W').sum()  # Недельная сумма
monthly_sales = df_ts['sales'].resample('ME').mean()  # Месячное среднее

print("📌 5.1 Ресемплинг по неделям:")
print(daily_sales.head(10))

print("\n📌 5.2 Скользящее среднее (7 дней):")
rolling_mean = df_ts['sales'].rolling(window=7).mean()
print(rolling_mean.head(10))

print("\n📌 5.3 Группировка по дню недели:")
weekday_stats = df_clean.groupby('weekday')['sales'].agg(['mean', 'median', 'count'])
print(weekday_stats)

# ==============================================================================
# ЧАСТЬ 6: ВИЗУАЛИЗАЦИЯ (ВСЕ ГРАФИКИ)
# ==============================================================================
print("\n" + "=" * 80)
print("📊 ЧАСТЬ 6: ПОСТРОЕНИЕ ГРАФИКОВ")
print("=" * 80)

# Создаем фигуру с несколькими подграфиками
fig = plt.figure(figsize=(16, 20))

# 6.1 Линейный график (временной ряд)
ax1 = plt.subplot(3, 3, 1)
df_ts['sales'].resample('W').sum().plot(ax=ax1, color='blue', linewidth=2)
ax1.set_title('Продажи по неделям (линейный график)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Дата')
ax1.set_ylabel('Сумма продаж')
ax1.grid(True, alpha=0.3)

# 6.2 Гистограмма распределения
ax2 = plt.subplot(3, 3, 2)
df_clean['sales'].hist(bins=30, ax=ax2, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(df_clean['sales'].mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {df_clean["sales"].mean():.0f}')
ax2.axvline(df_clean['sales'].median(), color='green', linestyle='--', linewidth=2, label=f'Медиана: {df_clean["sales"].median():.0f}')
ax2.set_title('Распределение продаж (гистограмма)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Продажи')
ax2.set_ylabel('Частота')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 Box plot по регионам
ax3 = plt.subplot(3, 3, 3)
df_clean.boxplot(column='sales', by='region', ax=ax3)
ax3.set_title('Распределение продаж по регионам (box plot)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Регион')
ax3.set_ylabel('Продажи')
plt.suptitle('')  # Убираем автоматический заголовок

# 6.4 Столбчатая диаграмма (продажи по продуктам)
ax4 = plt.subplot(3, 3, 4)
product_sales = df_clean.groupby('product')['sales'].mean().sort_values()
product_sales.plot(kind='barh', ax=ax4, color='coral')
ax4.set_title('Средние продажи по продуктам (bar chart)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Средние продажи')
ax4.set_ylabel('Продукт')
ax4.grid(True, alpha=0.3, axis='x')

# 6.5 Круговая диаграмма
ax5 = plt.subplot(3, 3, 5)
region_revenue = df_clean.groupby('region')['revenue'].sum()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
ax5.pie(region_revenue, labels=region_revenue.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax5.set_title('Доля выручки по регионам (pie chart)', fontsize=12, fontweight='bold')

# 6.6 Scatter plot (возраст vs продажи)
ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(df_clean['customer_age'], df_clean['sales'], 
                      c=df_clean['rating'], cmap='viridis', alpha=0.6, s=50)
ax6.set_title('Зависимость продаж от возраста покупателя (scatter)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Возраст')
ax6.set_ylabel('Продажи')
plt.colorbar(scatter, ax=ax6, label='Рейтинг')
ax6.grid(True, alpha=0.3)

# 6.7 Тепловая карта корреляции
ax7 = plt.subplot(3, 3, 7)
numeric_cols = ['sales', 'quantity', 'customer_age', 'rating', 'revenue', 'profit']
corr_matrix = df_clean[numeric_cols].corr()
im = ax7.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
ax7.set_xticks(range(len(numeric_cols)))
ax7.set_yticks(range(len(numeric_cols)))
ax7.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax7.set_yticklabels(numeric_cols)
ax7.set_title('Корреляционная матрица (heatmap)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax7)

# 6.8 Area chart (накопленные продажи по месяцам)
ax8 = plt.subplot(3, 3, 8)
monthly_by_product = pd.pivot_table(df_clean, values='sales', index='month', 
                                    columns='product', aggfunc='sum')
monthly_by_product.plot(kind='area', stacked=True, ax=ax8, alpha=0.7)
ax8.set_title('Накопленные продажи по месяцам (area chart)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Месяц')
ax8.set_ylabel('Сумма продаж')
ax8.legend(loc='upper left', fontsize=8)
ax8.grid(True, alpha=0.3)

# 6.9 Violin plot (распределение рейтинга по продуктам)
ax9 = plt.subplot(3, 3, 9)
sns.violinplot(data=df_clean, x='product', y='rating', ax=ax9, palette='Set2')
ax9.set_title('Распределение рейтинга по продуктам (violin plot)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Продукт')
ax9.set_ylabel('Рейтинг')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pandas_mega_cheatsheet_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Все графики сохранены в 'pandas_mega_cheatsheet_plots.png'")

# ==============================================================================
# ЧАСТЬ 7: ДОПОЛНИТЕЛЬНЫЕ ВОЗМОЖНОСТИ PANDAS
# ==============================================================================
print("\n" + "=" * 80)
print("🎯 ЧАСТЬ 7: ПРОДВИНУТЫЕ ФУНКЦИИ")
print("=" * 80)

# 7.1 Apply и лямбда-функции
print("\n📌 7.1 Использование apply:")
df_clean['sales_category'] = df_clean['sales'].apply(lambda x: 'High' if x > 1200 else 'Medium' if x > 800 else 'Low')
print(df_clean[['sales', 'sales_category']].head())

# 7.2 Query для фильтрации
print("\n📌 7.2 Использование query:")
high_sales = df_clean.query('sales > 1500 and region == "Север" and product == "Ноутбук"')
print(f"Найдено записей через query: {len(high_sales)}")

# 7.3 Pivot table с несколькими агрегациями
print("\n📌 7.3 Сложная сводная таблица:")
complex_pivot = pd.pivot_table(df_clean, 
                               values=['sales', 'quantity', 'rating'],
                               index='region',
                               columns='product',
                               aggfunc={'sales': 'mean', 'quantity': 'sum', 'rating': 'median'},
                               fill_value=0)
print(complex_pivot.head())

# 7.4 Экспорт в разные форматы
print("\n📌 7.4 Экспорт данных:")
df_clean.to_csv('pandas_export.csv', index=False)
df_clean.to_excel('pandas_export.xlsx', index=False, sheet_name='Sales Data')
print("✅ Данные экспортированы в CSV и Excel")

# 7.5 Чтение из файлов (демонстрация)
print("\n📌 7.5 Чтение сохраненных данных:")
df_read_csv = pd.read_csv('pandas_export.csv')
print(f"CSV прочитан: {df_read_csv.shape}")
df_read_excel = pd.read_excel('pandas_export.xlsx', sheet_name='Sales Data')
print(f"Excel прочитан: {df_read_excel.shape}")

# ==============================================================================
# ЧАСТЬ 8: ИТОГОВАЯ СТАТИСТИКА И ВЫВОДЫ
# ==============================================================================
print("\n" + "=" * 80)
print("📈 ЧАСТЬ 8: ИТОГОВЫЙ АНАЛИЗ")
print("=" * 80)

print("\n🏆 ТОП-10 лучших продаж:")
print(df_clean.nlargest(10, 'sales')[['date', 'region', 'product', 'sales', 'rating']])

print("\n📉 ТОП-10 худших продаж:")
print(df_clean.nsmallest(10, 'sales')[['date', 'region', 'product', 'sales', 'rating']])

print("\n💰 Общая статистика по выручке:")
print(f"Общая выручка: {df_clean['revenue'].sum():,.2f}")
print(f"Средняя выручка: {df_clean['revenue'].mean():,.2f}")
print(f"Медианная выручка: {df_clean['revenue'].median():,.2f}")
print(f"Максимальная выручка: {df_clean['revenue'].max():,.2f}")

print("\n📊 Статистика по возрастным группам:")
age_stats = df_clean.groupby('age_group').agg({
    'sales': 'mean',
    'revenue': 'sum',
    'customer_age': 'count'
}).rename(columns={'customer_age': 'count'})
print(age_stats)

# ==============================================================================
# ВЫВОД ИНФОРМАЦИИ О ВЫПОЛНЕНИИ
# ==============================================================================
print("\n" + "=" * 80)
print("✅ ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("=" * 80)
print("\n📁 Созданные файлы:")
print("   • pandas_mega_cheatsheet_plots.png - все графики")
print("   • pandas_export.csv - данные в CSV")
print("   • pandas_export.xlsx - данные в Excel")
print("\n💡 Советы по использованию шпоры:")
print("   1. Изменяйте параметры генерации данных для своих тестов")
print("   2. Экспериментируйте с разными типами графиков")
print("   3. Добавляйте свои операции в код")
print("   4. Используйте df.info() и df.describe() для быстрого анализа")
print("\n🎉 Шпора готова к использованию!")

# Очистка временных файлов (опционально)
# import os
# os.remove('pandas_export.csv')
# os.remove('pandas_export.xlsx')