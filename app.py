import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture # Используем GMM для более умной кластеризации
import math
import re
import random
import streamlit as st
import io

# --- Константы и настройки ---
WHISPER_MODEL = "medium"
OUTPUT_DIR = "./SpeechViz3D"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэшированные функции ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data
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

# --- Вспомогательные функции анализа ---
def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя' and (i == 0 or text_clean[i-1] not in 'аоуэыиьъ'):
            mapping = {'е': ['й', 'э'], 'ё': ['й', 'о'], 'ю': ['й', 'у'], 'я': ['й', 'а']}
            phonemes.extend(mapping[char])
        elif char in 'еёюя':
            mapping = {'е': 'э', 'ё': 'о', 'ю': 'у', 'я': 'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи':
            phonemes.append(char)
    return phonemes

def find_acoustic_features(formant_obj, pitch_obj, intensity_obj, start, end):
    f1_v, f2_v, p_v, i_v = [], [], [], []
    t = start
    while t < end:
        f1, f2 = formant_obj.get_value_at_time(1, t), formant_obj.get_value_at_time(2, t)
        p, i = pitch_obj.get_value_at_time(t), intensity_obj.get_value(t)
        if not math.isnan(f1): f1_v.append(f1)
        if not math.isnan(f2): f2_v.append(f2)
        if not math.isnan(p): p_v.append(p)
        if not math.isnan(i): i_v.append(i)
        t += 0.005
    
    return (np.nanmedian(f1_v) if f1_v else np.nan,
            np.nanmedian(f2_v) if f2_v else np.nan,
            end - start,
            np.nanmedian(p_v) if p_v else np.nan,
            np.nanmedian(i_v) if i_v else np.nan)

def analyze_vowel_segments(audio_path, transcription_segments):
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
        f_obj = sound.to_formant_burg()
        p_obj = sound.to_pitch()
        i_obj = sound.to_intensity()
    except: return []

    for seg in transcription_segments:
        phons = extract_phonemes(seg['word'])
        if not phons: continue
        
        v_only = [p for p in phons if p != 'й']
        if not v_only: continue
        
        dur_part = (seg['end'] - seg['start']) / len(v_only)
        curr = seg['start']
        
        for p in phons:
            if p == 'й':
                curr += 0.04
                continue
            f1, f2, d, pitch, intens = find_acoustic_features(f_obj, p_obj, i_obj, curr, curr + dur_part)
            if not np.isnan(f1) and not np.isnan(f2):
                energy = 0.00012 * (pitch * d) - 0.00015 if not np.isnan(pitch) else 0
                vowel_data.append({
                    'word': seg['word'], 'vowel': p, 'F1': f1, 'F2': f2,
                    'duration': d, 'mean_pitch': pitch, 'mean_intensity': intens,
                    'total_energy': energy
                })
            curr += dur_part
    return vowel_data

def normalize_lobanov(df):
    for col in ['F1', 'F2']:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / (df[col].std() or 1)
    return df

# --- Визуализация ---

def plot_3d_vowel_count(vowel_data, audio_filename):
    df = pd.DataFrame(vowel_data)
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    agg_list = []
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if not v_df.empty:
            agg_list.append({
                'vowel': v,
                'F1': v_df['F1'].mean(),
                'F2': v_df['F2'].mean(),
                'count': len(v_df),
                'energy': v_df['total_energy'].mean(),
                'pitch': v_df['mean_pitch'].mean()
            })
    
    plot_df = pd.DataFrame(agg_list)
    if plot_df.empty: return None

    fig = go.Figure()

    # Линии от пола до точек
    for _, row in plot_df.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['F1'], row['F1']], y=[row['F2'], row['F2']], z=[0, row['count']],
            mode='lines', line=dict(color='rgba(100,100,100,0.5)', width=4),
            showlegend=False, hoverinfo='none'
        ))

    # Основные точки
    fig.add_trace(go.Scatter3d(
        x=plot_df['F1'], y=plot_df['F2'], z=plot_df['count'],
        mode='markers+text',
        marker=dict(size=10, color=plot_df['count'], colorscale='Viridis', opacity=0.9),
        text=plot_df['vowel'],
        name='Гласные',
        hovertemplate=(
            "<b>Фонема: '%{text}'</b><br>" +
            "F1: %{x:.1f} Гц<br>" +
            "F2: %{y:.1f} Гц<br>" +
            "Кол-во упоминаний: %{z}<br>" +
            "Ср. энергия: %{customdata[0]:.6f}<br>" +
            "Ср. тон (Pitch): %{customdata[1]:.1f} Гц<extra></extra>"
        ),
        customdata=plot_df[['energy', 'pitch']]
    ))

    # Замыкающая линия и-ы-у-о-а-э-и
    line_df = pd.concat([plot_df, plot_df.iloc[[0]]])
    fig.add_trace(go.Scatter3d(
        x=line_df['F1'], y=line_df['F2'], z=line_df['count'],
        mode='lines', line=dict(color='red', width=6),
        name='Цепочка гласных', hoverinfo='none'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='F1 (Гц)', autorange="reversed"),
            yaxis=dict(title='F2 (Гц)', autorange="reversed"),
            zaxis=dict(title='Количество')
        ),
        width=1000, height=800
    )
    return fig

def plot_gmm_clustering(vowel_data):
    """Кластеризация методом GMM (Gaussian Mixture Model)"""
    df = pd.DataFrame(vowel_data)
    if len(df) < 6: return None
    
    df_norm = normalize_lobanov(df.copy())
    X = df_norm[['F1_z', 'F2_z']].values
    
    # Используем GMM для вероятностной кластеризации
    gmm = GaussianMixture(n_components=min(6, len(df)), random_state=42)
    df['cluster'] = gmm.fit_predict(X)
    
    fig = px.scatter(df, x='F1', y='F2', color='cluster', symbol='vowel',
                     title="GMM Кластеризация гласных (F1/F2)",
                     hover_data=['word', 'duration'],
                     color_continuous_scale='Turbo')
    
    fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))
    return fig

# --- Main App ---
def main():
    st.set_page_config(page_title="SpeechViz 2026", layout="wide")
    st.title("Аналитическая визуализация фонем")

    uploaded_file = st.file_uploader("Загрузите WAV", type=["wav"])
    
    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, "temp.wav")
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Процессинг
        segments = transcribe_cached(audio_path)
        vowel_data = analyze_vowel_segments(audio_path, segments)

        if vowel_data:
            # --- Вкладка 1: 3D Карта ---
            st.subheader("3D Карта")
            # НОВАЯ СТРОЧКА:
            st.info(f"✅ Анализ завершен. Всего определенных гласных фонем: **{len(vowel_data)}**")
            
            fig_3d = plot_3d_vowel_count(vowel_data, uploaded_file.name)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)

            st.divider()

            # --- Вкладка 2: Кластеризация ---
            st.subheader("Продвинутая кластеризация (GMM)")
            st.write("В отличие от K-means, этот метод ищет эллиптические скопления звуков, что точнее отражает артикуляцию.")
            fig_cl = plot_gmm_clustering(vowel_data)
            if fig_cl:
                st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.error("Не удалось извлечь достаточно данных.")

if __name__ == "__main__":
    main()
