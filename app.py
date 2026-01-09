import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
import math
import re
import random
import streamlit as st

# --- Константы и настройки ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- КЭШИРОВАННЫЕ ФУНКЦИИ (Ускорение) ---

@st.cache_resource(show_spinner="Загрузка Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = []
    for segment in segments:
        for word in segment.words:
            if word.probability > 0.1:
                word_level_segments.append({
                    'word': word.word.strip().lower(),
                    'start': word.start,
                    'end': word.end
                })
    return word_level_segments

@st.cache_data(show_spinner="Акустический анализ...")
def analyze_vowels_cached(audio_path, transcription_segments):
    return analyze_vowel_segments(audio_path, transcription_segments)

@st.cache_data(show_spinner=False)
def get_plot_3d_cached(vowel_data, audio_filename):
    return plot_3d_vowel_count(vowel_data, audio_filename)

@st.cache_data(show_spinner=False)
def get_radar_plot_cached(vowel_data, audio_filename, gender):
    return plot_radar_vowel_star(vowel_data, audio_filename, gender)

@st.cache_data(show_spinner="Кластеризация...")
def get_kmeans_plot_cached(vowel_data, audio_filename):
    return plot_kmeans_formant_map(vowel_data, audio_filename)

# --- Логика анализа ---

def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя':
            # Обработка йотированных
            if i == 0 or text_clean[i-1] in 'аоуэыиьъ':
                phonemes.extend(['й', 'э' if char=='е' else 'о' if char=='ё' else 'у' if char=='ю' else 'а'])
            else:
                phonemes.append('э' if char=='е' else 'о' if char=='ё' else 'у' if char=='ю' else 'а')
        elif char in 'аоуэыи':
            phonemes.append(char)
    return phonemes

def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
        formant_obj = sound.to_formant_burg()
        pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity()
    except Exception as e:
        st.error(f"Ошибка Praat: {e}")
        return []

    for seg in transcription_segments:
        word, w_start, w_end = seg['word'], seg['start'], seg['end']
        phonemes = extract_phonemes(word)
        vowels_only = [p for p in phonemes if p != 'й']
        if not vowels_only: continue
        
        # Распределяем время внутри слова
        total_j_time = phonemes.count('й') * J_DURATION
        v_dur = (w_end - w_start - total_j_time) / len(vowels_only)
        if v_dur <= 0: v_dur = 0.05

        curr_t = w_start
        for p in phonemes:
            if p == 'й':
                curr_t += J_DURATION
                continue
            
            mid_t = curr_t + v_dur/2
            f1 = formant_obj.get_value_at_time(1, mid_t)
            f2 = formant_obj.get_value_at_time(2, mid_t)
            pitch = pitch_obj.get_value_at_time(mid_t)
            intensity = intensity_obj.get_value(mid_t)
            
            if not (math.isnan(f1) or math.isnan(f2)):
                energy = (0.00012 * (pitch * v_dur if not math.isnan(pitch) else 0)) - 0.00015
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': f1, 'F2': f2,
                    'duration': v_dur, 'mean_pitch': pitch if not math.isnan(pitch) else 120,
                    'mean_intensity': intensity, 'total_energy': max(energy, 0.0001)
                })
            curr_t += v_dur
            
    return vowel_data

def get_russian_norms(gender='женщина'):
    if gender == 'мужчина':
        return {
            'и': {'F1': 290, 'F2': 2150, 'dur': 0.075, 'F0': 125},
            'ы': {'F1': 420, 'F2': 1350, 'dur': 0.080, 'F0': 120},
            'у': {'F1': 320, 'F2': 820,  'dur': 0.088, 'F0': 115},
            'о': {'F1': 460, 'F2': 920,  'dur': 0.092, 'F0': 118},
            'а': {'F1': 690, 'F2': 1300, 'dur': 0.108, 'F0': 115},
            'э': {'F1': 490, 'F2': 1750, 'dur': 0.082, 'F0': 122},
        }
    return {
        'и': {'F1': 320, 'F2': 2250, 'dur': 0.078, 'F0': 215},
        'ы': {'F1': 450, 'F2': 1400, 'dur': 0.082, 'F0': 205},
        'у': {'F1': 340, 'F2': 850,  'dur': 0.090, 'F0': 195},
        'о': {'F1': 480, 'F2': 950,  'dur': 0.095, 'F0': 200},
        'а': {'F1': 720, 'F2': 1350, 'dur': 0.110, 'F0': 195},
        'э': {'F1': 520, 'F2': 1850, 'dur': 0.085, 'F0': 210},
    }

# --- Визуализация ---

def plot_3d_vowel_count(vowel_data, audio_filename):
    df = pd.DataFrame(vowel_data)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    agg = df.groupby('vowel').mean(numeric_only=True).reindex(v_order)
    counts = df['vowel'].value_counts().reindex(v_order).fillna(0)
    
    fig = go.Figure()
    # Линии
    x, y, z = agg['F1'].tolist(), agg['F2'].tolist(), counts.tolist()
    # Замыкаем цикл
    x.append(x[0]); y.append(y[0]); z.append(z[0])
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', 
                               line=dict(color='red', width=6), name='Поток'))
    
    fig.update_layout(scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='Кол-во',
                                 xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")),
                      width=900, height=700)
    return fig, agg

def plot_radar_vowel_star(vowel_data, audio_filename, gender):
    df = pd.DataFrame(vowel_data)
    norms = get_russian_norms(gender)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    agg = df.groupby('vowel').mean(numeric_only=True).reindex(v_order)
    
    fig = go.Figure()
    for v in v_order:
        if v not in agg.index or pd.isna(agg.loc[v, 'F1']): continue
        p, n = agg.loc[v], norms[v]
        vals = [
            (p['F1']-n['F1'])/n['F1']*100, (p['F2']-n['F2'])/n['F2']*100,
            (p['duration']-n['dur'])/n['dur']*100, 12*np.log2(p['mean_pitch']/n['F0'])
        ]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], 
                                      theta=['F1%','F2%','Dur%','Pitch','F1%'], 
                                      fill='toself', name=v))
    
    fig.update_layout(polar=dict(radialaxis=dict(range=[-100, 100])), showlegend=True)
    return fig, agg

def plot_kmeans_formant_map(vowel_data, audio_filename):
    from sklearn.cluster import KMeans
    df = pd.DataFrame(vowel_data)
    # Нормализация Лобанова
    df['F1_z'] = (df['F1'] - df['F1'].mean()) / df['F1'].std()
    df['F2_z'] = (df['F2'] - df['F2'].mean()) / df['F2'].std()
    
    km = KMeans(n_clusters=min(len(df), 6), random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(df[['F1_z', 'F2_z']])
    
    fig = px.scatter(df, x='F1', y='F2', color='cluster', text='vowel', 
                     title="Кластеризация K-means")
    fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))
    return fig

# --- Основное приложение ---

def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D Pro")
    st.title("SpeechViz3D: Профессиональный анализ")

    up = st.file_uploader("WAV файл", type="wav")
    if up:
        path = os.path.join(OUTPUT_DIR, up.name)
        with open(path, "wb") as f: f.write(up.getbuffer())

        # 1. Анализ (кэшируется)
        segments = transcribe_cached(path)
        vowel_data = analyze_vowels_cached(path, segments)
        
        if not vowel_data:
            st.error("Гласные не распознаны."); return

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 3D График
            st.subheader("3D Динамика")
            fig3, _ = get_plot_3d_cached(vowel_data, up.name)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Кластеризация
            st.subheader("Кластерный анализ")
            fig_km = get_kmeans_plot_cached(vowel_data, up.name)
            st.plotly_chart(fig_km, use_container_width=True)

        with col2:
            # Звезда (быстрое переключение)
            st.subheader("Сравнение с нормой")
            gender = st.radio("Пол пациента:", ["женщина", "мужчина"])
            fig_r, radar_df = get_radar_plot_cached(vowel_data, up.name, gender)
            st.plotly_chart(fig_r, use_container_width=True)
            
            st.download_button("Скачать данные (CSV)", 
                               pd.DataFrame(vowel_data).to_csv(index=False).encode('utf-8'),
                               "results.csv", "text/csv")

if __name__ == "__main__":
    main()
