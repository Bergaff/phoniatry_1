import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
import math
import re
import streamlit as st

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL_NAME = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Карта цветов для фонем
VOWEL_COLORS = {
    'а': '#FF4B4B', 'о': '#FFA500', 'у': '#800080', 
    'ы': '#00FF00', 'и': '#0000FF', 'э': '#00CED1'
}

# --- Кэширование ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL_NAME, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    return [{'word': w.word.strip().lower(), 'start': w.start, 'end': w.end} 
            for s in segments for w in s.words if w.probability > 0.1]

# --- Анализ ---
def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя':
            if i == 0 or text_clean[i-1] not in 'аоуэыиьъ': phonemes.append('й')
            mapping = {'е':'э', 'ё':'о', 'ю':'у', 'я':'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def analyze_vowel_segments(audio_path, segments):
    sound = parselmouth.Sound(audio_path)
    f_obj = sound.to_formant_burg()
    p_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR)
    i_obj = sound.to_intensity()
    
    results = []
    for seg in segments:
        phons = extract_phonemes(seg['word'])
        if not phons: continue
        dur_each = (seg['end'] - seg['start']) / len(phons)
        curr = seg['start']
        for p in phons:
            if p == 'й': 
                curr += 0.04
                continue
            mid = curr + dur_each / 2
            f1, f2 = f_obj.get_value_at_time(1, mid), f_obj.get_value_at_time(2, mid)
            pitch = p_obj.get_value_at_time(mid)
            intensity = i_obj.get_value(mid)
            if not any(math.isnan(v) for v in [f1, f2, intensity]):
                results.append({
                    'vowel': p, 'F1': f1, 'F2': f2, 
                    'intensity': intensity, 'pitch': pitch, 'word': seg['word']
                })
            curr += dur_each
    return pd.DataFrame(results)

# --- Визуализация ---
def plot_vowel_3d_spheres(df):
    # Нормализация размера шаров (интенсивность -> размер)
    df['size'] = (df['intensity'] - df['intensity'].min()) / (df['intensity'].max() - df['intensity'].min() + 1) * 30 + 10
    
    fig = px.scatter_3d(
        df, x='F1', y='F2', z=[0]*len(df), # Все лежат в основании (Z=0)
        color='vowel',
        size='size',
        color_discrete_map=VOWEL_COLORS,
        labels={'z': 'Основание'},
        hover_data=['word', 'intensity', 'F1', 'F2']
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(autorange="reversed", title='F1 (Гц)'),
            yaxis=dict(autorange="reversed", title='F2 (Гц)'),
            zaxis=dict(title='', showticklabels=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Карта гласных (Размер = Интенсивность, Z=0)"
    )
    return fig

def plot_radar(df):
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    avg = df.groupby('vowel')[['F1', 'F2', 'intensity']].mean().reindex(vowel_order).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg['F1'].values.tolist() + [avg['F1'].iloc[0]],
        theta=vowel_order + [vowel_order[0]],
        fill='toself', name='Средний F1'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Акустический профиль (F1)")
    return fig

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="VowelViz 3D")
    st.title("🎤 Анализ артикуляции: 3D Карта гласных")

    up_file = st.file_uploader("Загрузите WAV файл", type=["wav"])
    
    if up_file:
        path = os.path.join(OUTPUT_DIR, up_file.name)
        with open(path, "wb") as f: f.write(up_file.getbuffer())
        
        # Обработка
        word_segs = transcribe_cached(path)
        df = analyze_vowel_segments(path, word_segs)
        
        if not df.empty:
            # ВКЛАДКИ
            tab1, tab2, tab3 = st.tabs(["📊 3D Карта", "📈 Гистограмма", "🌟 Звезда гласных"])
            
            with tab1:
                st.plotly_chart(plot_vowel_3d_spheres(df), use_container_width=True)
                st.info("Размер шара зависит от громкости (интенсивности) произношения.")
            
            with tab2:
                fig_hist = px.histogram(df, x="vowel", color="vowel", 
                                      color_discrete_map=VOWEL_COLORS, title="Частота появления фонем")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with tab3:
                st.plotly_chart(plot_radar(df), use_container_width=True)
        else:
            st.error("Гласные не обнаружены. Попробуйте другую запись.")

if __name__ == "__main__":
    main()
