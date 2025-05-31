import pandas as pd

# Путь к файлу (измени при необходимости)
CSV_PATH = "RuATD/human_baseline.csv"

def main():
    # Загружаем данные
    df = pd.read_csv(CSV_PATH)

    # Проверим названия колонок
    print(f"[INFO] Столбцы в датасете: {df.columns.tolist()}")

    # Приводим названия к стандартному виду (если нужно)
    df = df.rename(columns={col: col.strip().lower() for col in df.columns})

    # Предположим, что нужная колонка — majority_vote
    if 'majority_vote' not in df.columns:
        print("[ERROR] Не найден столбец 'majority_vote'")
        return

    total = len(df)
    counts = df['majority_vote'].value_counts()

    print(f"\n[СТАТИСТИКА ДАТАСЕТА]")
    print(f"Всего данных: {total}")
    print(f"Класс 'Human' (H): {counts.get('H', 0)}")
    print(f"Класс 'AI' (M): {counts.get('M', 0)}")

if __name__ == "__main__":
    main()
