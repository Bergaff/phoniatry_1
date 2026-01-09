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
import io
import streamlit.components.v1 as components
from sklearn.cluster import KMeans

# --- Кэширование ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация аудио...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = []
    full_text_list = []
    for segment in segments:
        for word in segment.words:
            if word.probability > 0.1:
                word_level_segments.append({
                    'word': word.word.strip().lower(),
                    'start': word.start,
                    'end': word.end
                })
                full_text_list.append(word.word.strip())
    return word_level_segments, " ".join(full_text_list)

@st.cache_data(show_spinner="Акустический анализ...")
def analyze_vowels_cached(audio_path, transcription_segments):
    return analyze_vowel_segments(audio_path, transcription_segments)

@st.cache_data(show_spinner=False)
def get_plot_3d(vowel_data, audio_filename):
    return plot_3d_vowel_count(vowel_data, audio_filename)

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def find_acoustic_features(formant_obj, pitch_obj, intensity_obj, segment_start, segment_end):
    F1_values, F2_values, pitch_values, intensity_values = [], [], [], []
    t = segment_start
    while t < segment_end:
        f1 = formant_obj.get_value_at_time(1, t)
        f2 = formant_obj.get_value_at_time(2, t)
        pitch = pitch_obj.get_value_at_time(t)
        intensity = intensity_obj.get_value(t)
        if not math.isnan(f1): F1_values.append(f1)
        if not math.isnan(f2): F2_values.append(f2)
        if not math.isnan(pitch): pitch_values.append(pitch)
        if not math.isnan(intensity): intensity_values.append(intensity)
        t += 0.005
    return (np.nanmedian(F1_values), np.nanmedian(F2_values), segment_end - segment_start, 
            np.nanmedian(pitch_values), np.nanmedian(intensity_values))

def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data, phoneme_log_data = [], []
    sound = parselmouth.Sound(audio_path)
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    intensity_obj = sound.to_intensity()

    for segment in transcription_segments:
        word = segment['word']
        phonemes_in_word = extract_phonemes(word)
        if not phonemes_in_word: continue
        
        vowels_only = [p for p in phonemes_in_word if p != 'й']
        if not vowels_only: continue
        
        dur_per_vowel = (segment['end'] - segment['start'] - (phonemes_in_word.count('й')*J_DURATION)) / len(vowels_only)
        current_time = segment['start']

        for phoneme in phonemes_in_word:
            if phoneme == 'й':
                current_time += J_DURATION
                continue
            v_start, v_end = current_time, current_time + dur_per_vowel
            f1, f2, dur, pitch, intens = find_acoustic_features(formant_obj, pitch_obj, intensity_obj, v_start, v_end)
            
            if not np.isnan([f1, f2, pitch]).any():
                total_energy = 0.00012 * (pitch * dur) - 0.00015
                entry = {
                    'word': word, 'vowel': phoneme, 'F1': f1, 'F2': f2,
                    'duration': dur, 'mean_pitch': pitch, 'mean_intensity': intens,
                    'total_energy': total_energy
                }
                vowel_data.append(entry)
                phoneme_log_data.append(entry)
            current_time = v_end
    return vowel_data, phoneme_log_data

def normalize_lobanov(df, cols=['F1', 'F2']):
    df_norm = df.copy()
    for col in cols:
        mean_v, std_v = df[col].mean(), df[col].std()
        df_norm[f'{col}_z'] = (df[col] - mean_v) / (std_v if std_v != 0 else 1)
    return df_norm

def plot_3d_vowel_count(vowel_data, audio_filename):
    if not vowel_data: return None, None
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    df = pd.DataFrame(vowel_data)
    
    # Агрегация данных для 3D пиков
    stats = []
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if not v_df.empty:
            stats.append({
                'vowel': v,
                'count': len(v_df),
                'avg_F1': v_df['F1'].mean(),
                'avg_F2': v_df['F2'].mean(),
                'avg_energy': v_df['total_energy'].mean(),
                'avg_pitch': v_df['mean_pitch'].mean()
            })
    
    plot_df = pd.DataFrame(stats)
    fig = go.Figure()

    # Линии от пола до пиков и сами пики
    for _, row in plot_df.iterrows():
        # Вертикальная линия
        fig.add_trace(go.Scatter3d(
            x=[row['avg_F1'], row['avg_F1']], y=[row['avg_F2'], row['avg_F2']], z=[0, row['count']],
            mode='lines', line=dict(color='gray', width=3), hoverinfo='none', showlegend=False
        ))
        
        # Точка на полу (Z=0) с исправленным Hover
        fig.add_trace(go.Scatter3d(
            x=[row['avg_F1']], y=[row['avg_F2']], z=[0],
            mode='markers+text',
            text=[f'"{row["vowel"]}"'], textposition="bottom center",
            marker=dict(size=10, color='blue', opacity=0.6),
            name=f'База {row["vowel"]}',
            hovertemplate = (
                f'<b>Фонема: "{row["vowel"]}"</b><br>' +
                f'F1: %{{x:.0f}} Гц<br>' +
                f'F2: %{{y:.0f}} Гц<br>' +
                f'Кол-во упоминаний: {row["count"]}<br>' +
                f'Средняя энергия: {row["avg_energy"]:.6f}<br>' +
                f'Средний Pitch: {row["avg_pitch"]:.1f} Гц<extra></extra>'
            )
        ))

    # Красная ломаная линия по пикам
    if len(plot_df) > 1:
        line_df = pd.concat([plot_df, plot_df.iloc[[0]]])
        fig.add_trace(go.Scatter3d(
            x=line_df['avg_F1'], y=line_df['avg_F2'], z=line_df['count'],
            mode='lines+markers', line=dict(color='red', width=5),
            marker=dict(size=4), name='Цепочка гласных', hoverinfo='none'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='F1', yaxis_title='F2', zaxis_title='Кол-во',
            xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")
        ),
        width=1000, height=800, title="3D Карта распределения гласных"
    )
    return fig, plot_df.to_dict('records')

def plot_clustering_hulls(vowel_data):
    df = pd.DataFrame(vowel_data)
    if len(df) < 6: return go.Figure()
    
    df_norm = normalize_lobanov(df, ['F1', 'F2'])
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df_norm[['F1_z', 'F2_z']])
    
    fig = go.Figure()
    colors = px.colors.qualitative.Safe

    for i in range(6):
        c_df = df[df['cluster'] == i]
        if len(c_df) < 3: continue
        
        # Область кластера (Convex Hull)
        points = c_df[['F1', 'F2']].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)
        
        fig.add_trace(go.Scatter(
            x=hull_points[:,0], y=hull_points[:,1], fill="toself",
            fillcolor=colors[i], opacity=0.2, line=dict(color=colors[i], width=1),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=c_df['F1'], y=c_df['F2'], mode='markers',
            marker=dict(color=colors[i], size=8),
            name=f'Кластер {i+1}',
            text=c_df['vowel'],
            hovertemplate='Фонема: %{text}<br>F1: %{x}<br>F2: %{y}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="F1 (Гц)", yaxis_title="F2 (Гц)",
        xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"),
        title="Кластеризация областей гласных (Convex Hull)"
    )
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Акустический анализ речи")

    uploaded_file = st.file_uploader("Загрузите WAV", type=["wav"])

    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Анализ
        segments, full_text = transcribe_cached(audio_path)
        vowel_data, _ = analyze_vowels_cached(audio_path, segments)

        if vowel_data:
            # Секция транскрибации
            st.subheader("Результат транскрибации")
            st.write(full_text)
            st.info(f"Определено гласных фонем: {len(vowel_data)}")

            tab1, tab2 = st.tabs(["3D Карта", "Кластеризация"])

            with tab1:
                fig3d, _ = get_plot_3d(vowel_data, audio_path)
                st.plotly_chart(fig3d, use_container_width=True)

            with tab2:
                st.subheader("Улучшенная кластеризация")
                fig_hull = plot_clustering_hulls(vowel_data)
                st.plotly_chart(fig_hull, use_container_width=True)

if __name__ == "__main__":
    main()
