import pandas as pd

CSV_PATH = "RuATD/human_baseline.csv"
CSV_TEST_PATH = "RuATD/test.csv"
CSV_TRAIN_PATH = "RuATD/train.csv"
CSV_VAL_PATH = "RuATD/val.csv"
CSV = "RuATD/sample_submit_multiple.csv"
def main():
    # загружаем данные
    df = pd.read_csv(CSV)

    # названия колонок
    print(f"[INFO] Столбцы в датасете: {df.columns.tolist()}")

    # названия к стандартному виду
    df = df.rename(columns={col: col.strip().lower() for col in df.columns})
    print(f"\n[СТАТИСТИКА ДАТАСЕТА]")
    print(f"Всего данных: {len(df)}")

    if 'class' not in df.columns:
        print("[ERROR] Не найден столбец 'class'")
        return

    total = len(df)
    counts = df['class'].value_counts()

    print(f"\n[СТАТИСТИКА ДАТАСЕТА]")
    print(f"Всего данных: {total}")
    print(f"Класс 'Human' (H): {counts.get('Human', 0)}")
    print(f"Класс 'AI' (M): {total - counts.get('Human', 0)}")

if __name__ == "__main__":
    main()
