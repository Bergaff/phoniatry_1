import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
from sklearn.mixture import GaussianMixture # GMM для улучшенной кластеризации
import math
import re
import random
import streamlit as st

# --- Кэширование ресурсов ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel("medium", device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    return [{'word': w.word.strip().lower(), 'start': w.start, 'end': w.end} 
            for s in segments for w in s.words if w.probability > 0.1]

@st.cache_data(show_spinner="Анализ...")
def analyze_vowels_cached(audio_path, transcription_segments):
    return analyze_vowel_segments(audio_path, transcription_segments)

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def analyze_vowel_segments(audio_path, transcription_segments):
    sound = parselmouth.Sound(audio_path)
    f0 = sound.to_pitch()
    formants = sound.to_formant_burg()
    intensity = sound.to_intensity()
    
    vowel_data = []
    for seg in transcription_segments:
        phons = extract_phonemes(seg['word'])
        if not phons: continue
        
        dur_per_vowel = (seg['end'] - seg['start']) / len(phons)
        curr = seg['start']
        
        for p in phons:
            if p == 'й': 
                curr += 0.04
                continue
            
            mid = curr + dur_per_vowel / 2
            f1 = formants.get_value_at_time(1, mid)
            f2 = formants.get_value_at_time(2, mid)
            pitch = f0.get_value_at_time(mid)
            inten = intensity.get_value(mid)
            
            if not any(math.isnan(x) for x in [f1, f2, pitch]):
                vowel_data.append({
                    'vowel': p, 'F1': f1, 'F2': f2, 'pitch': pitch, 
                    'intensity': inten, 'duration': dur_per_vowel,
                    'energy': 0.00012 * (pitch * dur_per_vowel)
                })
            curr += dur_per_vowel
    return vowel_data

def normalize_lobanov(df):
    for col in ['F1', 'F2']:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    return df

# --- 1. УЛУЧШЕННАЯ 3D КАРТА ---
def plot_3d_vowel_map(vowel_data, audio_filename):
    df = pd.DataFrame(vowel_data)
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    # Агрегация данных
    summary = []
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if not v_df.empty:
            summary.append({
                'vowel': v,
                'F1': v_df['F1'].mean(),
                'F2': v_df['F2'].mean(),
                'count': len(v_df),
                'energy': v_df['energy'].mean(),
                'jitter': v_df['pitch'].std() / v_df['pitch'].mean() if len(v_df)>1 else 0
            })
    
    df_sum = pd.DataFrame(summary)
    if df_sum.empty: return None

    fig = go.Figure()

    # Линии от пола до точек
    for _, row in df_sum.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['F1'], row['F1']], y=[row['F2'], row['F2']], z=[0, row['count']],
            mode='lines', line=dict(color='rgba(100,100,100,0.5)', width=4),
            showlegend=False, hoverinfo='none'
        ))

    # Основные точки
    fig.add_trace(go.Scatter3d(
        x=df_sum['F1'], y=df_sum['F2'], z=df_sum['count'],
        mode='markers+text',
        marker=dict(size=12, color=df_sum['count'], colorscale='Viridis', opacity=0.9),
        text=[f'"{v}"' for v in df_sum['vowel']],
        textposition="top center",
        hovertemplate=(
            "<b>Фонема: \"%{customdata[0]}\"</b><br>" +
            "F1: %{x:.1f} Гц<br>" +
            "F2: %{y:.1f} Гц<br>" +
            "Количество упоминаний: %{z}<br>" +
            "Средняя энергия: %{customdata[1]:.6f}<br>" +
            "Джиттер (approx): %{customdata[2]:.4f}<br>" +
            "<extra></extra>"
        ),
        customdata=np.stack((df_sum['vowel'], df_sum['energy'], df_sum['jitter']), axis=-1),
        name='Гласные'
    ))

    # Соединительная линия (траектория)
    fig.add_trace(go.Scatter3d(
        x=df_sum['F1'].tolist() + [df_sum['F1'].iloc[0]],
        y=df_sum['F2'].tolist() + [df_sum['F2'].iloc[0]],
        z=df_sum['count'].tolist() + [df_sum['count'].iloc[0]],
        mode='lines', line=dict(color='red', width=2), name='Цепочка фонем',
        hoverinfo='none'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='F1 (Гц)', yaxis_title='F2 (Гц)', zaxis_title='Количество',
            xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000, height=800
    )
    return fig

# --- 2. НОВАЯ КЛАСТЕРИЗАЦИЯ (GMM) ---
def plot_gmm_clustering(vowel_data):
    df = pd.DataFrame(vowel_data)
    if len(df) < 6: return None
    
    df_norm = normalize_lobanov(df.copy())
    X = df_norm[['F1_z', 'F2_z']].values
    
    # Используем GMM (Gaussian Mixture Model) - она круче K-means для речи
    gmm = GaussianMixture(n_components=6, random_state=42)
    df_norm['cluster'] = gmm.fit_predict(X)
    
    fig = px.scatter(df_norm, x='F1', y='F2', color='vowel', 
                     symbol='cluster', 
                     title="Кластеризация GMM (Гауссовы смеси) в пространстве F1-F2",
                     hover_data=['energy', 'duration'])
    
    fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))
    return fig

# --- Main App ---
def main():
    st.set_page_config(page_title="SpeechViz3D Pro", layout="wide")
    st.title("Акустический анализ речи")
    
    file = st.file_uploader("Загрузите WAV", type=['wav'])
    if file:
        path = os.path.join(OUTPUT_DIR, file.name)
        with open(path, "wb") as f: f.write(file.getbuffer())
        
        segments = transcribe_cached(path)
        vowel_data = analyze_vowels_cached(path, segments)
        
        tab1, tab2 = st.tabs(["📊 3D Карта гласных", "🧬 Продвинутая кластеризация"])
        
        with tab1:
            st.subheader("3D визуализация формантного пространства")
            # НОВАЯ СТРОЧКА:
            st.info(f"Общее количество определенных гласных фонем: **{len(vowel_data)}**")
            
            fig3 = plot_3d_vowel_map(vowel_data, file.name)
            if fig3: st.plotly_chart(fig3, use_container_width=True)
            
        with tab2:
            st.subheader("Кластеризация методом Гауссовых смесей (GMM)")
            st.write("В отличие от K-means, GMM учитывает эллиптическую форму распределения формант.")
            fig_cluster = plot_gmm_clustering(vowel_data)
            if fig_cluster: st.plotly_chart(fig_cluster, use_container_width=True)

if __name__ == "__main__":
    main()
