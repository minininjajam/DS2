import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def prepare():
    print("Начало очистки данных...")
    try:
        df = pd.read_csv('dataset.csv')
    except UnicodeDecodeError:
        df = pd.read_csv('dataset.csv', encoding='cp1251')

    # Очистка
    df = df.dropna(subset=['track_name', 'artists', 'track_genre'])
    df = df.drop_duplicates(subset=['track_name', 'artists'])

    # Ограничиваем до 15к чтобы бот быстрее обрабатывал данные
    df_small = df.head(15000).copy()

    df_small['vibe_text'] = (df_small['track_genre'].astype(str).str.lower() + " ") * 2 + \
                            df_small['track_name'].astype(str).str.lower() + " by " + \
                            df_small['artists'].astype(str).str.lower()

    print("Создание эмбеддингов ")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(df_small['vibe_text'].tolist(), show_progress_bar=True)

    np.save('music_embeddings.npy', embeddings.astype('float32'))
    df_small.to_csv('processed_music.csv', index=False)
    print("Файлы processed_music.csv и music_embeddings.npy созданы!")


if __name__ == "__main__":
    prepare()
