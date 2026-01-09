import os
import parselmouth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from faster_whisper import WhisperModel
from scipy.spatial import ConvexHull
import math
import re
import random
import streamlit as st
from sklearn.cluster import KMeans

# --- Константы и Инициализация ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = []
    full_text = []
    for segment in segments:
        for word in segment.words:
            if word.probability > 0.1:
                word_level_segments.append({'word': word.word.strip().lower(), 'start': word.start, 'end': word.end})
                full_text.append(word.word.strip())
    return word_level_segments, " ".join(full_text)

def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя' and (i == 0 or text_clean[i-1] not in 'аоуэыиьъ'):
            if char == 'е': phonemes.extend(['й', 'э'])
            elif char == 'ё': phonemes.extend(['й', 'о'])
            elif char == 'ю': phonemes.extend(['й', 'у'])
            elif char == 'я': phonemes.extend(['й', 'а'])
        elif char in 'еёюя':
            if char == 'е': phonemes.append('э')
            elif char == 'ё': phonemes.append('о')
            elif char == 'ю': phonemes.append('у')
            elif char == 'я': phonemes.append('а')
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def find_acoustic_features(formant_obj, pitch_obj, intensity_obj, start, end):
    f1_l, f2_l, p_l, i_l = [], [], [], []
    t = start
    while t < end:
        f1, f2 = formant_obj.get_value_at_time(1, t), formant_obj.get_value_at_time(2, t)
        pitch, inten = pitch_obj.get_value_at_time(t), intensity_obj.get_value(t)
        if not math.isnan(f1): f1_l.append(f1)
        if not math.isnan(f2): f2_l.append(f2)
        if not math.isnan(pitch): p_l.append(pitch)
        if not math.isnan(inten): i_l.append(inten)
        t += 0.005
    return np.nanmedian(f1_l), np.nanmedian(f2_l), end-start, np.nanmedian(p_l), np.nanmedian(i_l)

def analyze_vowel_segments(audio_path, transcription_segments):
    sound = parselmouth.Sound(audio_path)
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    intensity_obj = sound.to_intensity()
    vowel_data = []
    for seg in transcription_segments:
        word = seg['word']
        phonemes = extract_phonemes(word)
        v_only = [p for p in phonemes if p != 'й']
        if not v_only: continue
        dur_v = (seg['end'] - seg['start'] - (phonemes.count('й')*0.04)) / len(v_only)
        curr = seg['start']
        for p in phonemes:
            if p == 'й': curr += 0.04; continue
            f1, f2, d, pitch, intens = find_acoustic_features(formant_obj, pitch_obj, intensity_obj, curr, curr + dur_v)
            if not np.isnan([f1, f2, pitch]).any():
                energy = 0.00012 * (pitch * d) - 0.00015
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': f1, 'F2': f2, 
                    'duration': d, 'mean_pitch': pitch, 'mean_intensity': intens, 'total_energy': energy
                })
            curr += dur_v
    return vowel_data

def plot_3d_vowel_count(vowel_data):
    df = pd.DataFrame(vowel_data)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    stats = []
    for v in v_order:
        v_df = df[df['vowel'] == v]
        if not v_df.empty:
            stats.append({
                'vowel': v, 'count': len(v_df), 'F1': v_df['F1'].mean(), 'F2': v_df['F2'].mean(),
                'energy': v_df['total_energy'].mean(), 'pitch': v_df['mean_pitch'].mean(),
                'intensity': v_df['mean_intensity'].mean()
            })
    
    pdf = pd.DataFrame(stats)
    fig = go.Figure()
    
    # Размер кружка от СРЕДНЕЙ ЭНЕРГИИ
    min_e, max_e = pdf['energy'].min(), pdf['energy'].max()
    # Нормализуем размер для визуализации (от 15 до 45 пикселей)
    if max_e != min_e:
        sizes = [15 + (e - min_e) / (max_e - min_e) * 30 for e in pdf['energy']]
    else:
        sizes = [25] * len(pdf)

    for i, row in pdf.iterrows():
        # Вертикальная линия
        fig.add_trace(go.Scatter3d(
            x=[row['F1'], row['F1']], y=[row['F2'], row['F2']], z=[0, row['count']],
            mode='lines', line=dict(color='gray', width=3), showlegend=False, hoverinfo='skip'
        ))
        # Сфера (размер от энергии)
        fig.add_trace(go.Scatter3d(
            x=[row['F1']], y=[row['F2']], z=[0],
            mode='markers+text', text=[f'"{row["vowel"]}"'], textposition="bottom center",
            marker=dict(size=sizes[i], color=i, colorscale='Viridis', opacity=0.8),
            name=f'Фонема "{row["vowel"]}"',
            hovertemplate=(
                f'<b>Фонема: "{row["vowel"]}"</b><br>'
                f'F1: %{{x:.0f}} Гц<br>F2: %{{y:.0f}} Гц<br>'
                f'Кол-во упоминаний: {row["count"]}<br>'
                f'Средняя энергия: {row["energy"]:.6f}<br>'
                f'Средний Pitch: {row["pitch"]:.1f} Гц<extra></extra>'
            )
        ))

    if len(pdf) > 1:
        ldf = pd.concat([pdf, pdf.iloc[[0]]])
        fig.add_trace(go.Scatter3d(x=ldf['F1'], y=ldf['F2'], z=ldf['count'], mode='lines', line=dict(color='red', width=4), name='Цепочка'))

    fig.update_layout(scene=dict(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")), width=1000, height=700)
    return fig, pdf

def plot_clustering_hulls(vowel_data):
    df = pd.DataFrame(vowel_data)
    if len(df) < 6: return go.Figure()
    df_n = df.copy()
    for c in ['F1', 'F2']:
        df_n[c] = (df[c] - df[c].mean()) / df[c].std()
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df_n[['F1', 'F2']])
    fig = go.Figure()
    colors = px.colors.qualitative.Safe
    for i in range(6):
        c_df = df[df['cluster'] == i]
        if len(c_df) < 3: continue
        points = c_df[['F1', 'F2']].values
        hull = ConvexHull(points)
        h_pts = points[hull.vertices]
        h_pts = np.append(h_pts, [h_pts[0]], axis=0)
        fig.add_trace(go.Scatter(x=h_pts[:,0], y=h_pts[:,1], fill="toself", fillcolor=colors[i], opacity=0.2, line=dict(color=colors[i]), showlegend=False))
        fig.add_trace(go.Scatter(x=c_df['F1'], y=c_df['F2'], mode='markers', marker=dict(color=colors[i], size=10), name=f'Кластер {i+1}', text=c_df['vowel']))
    fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"), title="Кластеризация (Области гласных)")
    return fig

def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D")
    st.title("Анализ гласных фонем")

    file = st.file_uploader("Загрузите WAV", type=["wav"])
    if file:
        path = os.path.join(OUTPUT_DIR, file.name)
        with open(path, "wb") as f: f.write(file.getbuffer())

        segments, text = transcribe_cached(path)
        vowel_data = analyze_vowel_segments(path, segments)

        if vowel_data:
            st.subheader("Транскрибация")
            st.write(text)
            st.success(f"Определено гласных фонем: {len(vowel_data)}")
            
            tabs = st.tabs(["3D Карта", "Гистограмма", "Кластеризация"])

            with tabs[0]:
                fig3d, summary_df = plot_3d_vowel_count(vowel_data)
                st.plotly_chart(fig3d, use_container_width=True)
                
                # КНОПКИ СКАЧИВАНИЯ ПОД ГРАФИКОМ
                col1, col2 = st.columns(2)
                with col1:
                    full_csv = pd.DataFrame(vowel_data).to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Скачать таблицу со всеми данными (все сегменты)", full_csv, "all_phonemes_data.csv", "text/csv")
                with col2:
                    summary_csv = summary_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Скачать усредненные данные (все параметры)", summary_csv, "averaged_vowel_params.csv", "text/csv")

            with tabs[1]:
                df_v = pd.DataFrame(vowel_data)
                fig_h = px.histogram(df_v, x='vowel', color='vowel', title="Частота появления фонем")
                st.plotly_chart(fig_h, use_container_width=True)

            with tabs[2]:
                st.plotly_chart(plot_clustering_hulls(vowel_data), use_container_width=True)

if __name__ == "__main__":
    main()
