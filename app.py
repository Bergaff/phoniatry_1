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

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэширование ресурсов ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация аудио...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = [{
        'word': word.word.strip().lower(),
        'start': word.start,
        'end': word.end
    } for segment in segments for word in segment.words if word.probability > 0.1]
    return word_level_segments

@st.cache_data(show_spinner="Акустический анализ (Praat + Микропараметры)...")
def analyze_vowels_cached(audio_path, transcription_segments):
    return analyze_vowel_segments(audio_path, transcription_segments)

# --- Вспомогательные функции ---
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

def find_acoustic_features(sound, formant_obj, pitch_obj, intensity_obj, segment_start, segment_end):
    """Извлекает F1, F2, F0, Интенсивность + Jitter/Shimmer/HNR."""
    # Обрезка звука для анализа микропараметров (PointProcess)
    vowel_part = sound.extract_part(from_time=segment_start, to_time=segment_end)
    point_process = parselmouth.praat.call(vowel_part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    
    # Стандартные параметры
    f1 = formant_obj.get_value_at_time(1, (segment_start + segment_end)/2)
    f2 = formant_obj.get_value_at_time(2, (segment_start + segment_end)/2)
    pitch = pitch_obj.get_value_at_time((segment_start + segment_end)/2)
    intensity = intensity_obj.get_value((segment_start + segment_end)/2)
    
    # Микропараметры (Джиттер, Шиммер, HNR)
    try:
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([vowel_part, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = sound.to_harmonicity()
        hnr = harmonicity.get_value((segment_start + segment_end)/2)
    except:
        jitter, shimmer, hnr = np.nan, np.nan, np.nan

    return f1, f2, (segment_end - segment_start), pitch, intensity, jitter, shimmer, hnr

def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data = []
    
    sound = parselmouth.Sound(audio_path)
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    intensity_obj = sound.to_intensity()

    for segment in transcription_segments:
        word, w_start, w_end = segment['word'], segment['start'], segment['end']
        phonemes = extract_phonemes(word)
        vowels_only = [p for p in phonemes if p != 'й']
        if not vowels_only: continue
        
        # Распределение времени
        j_count = phonemes.count('й')
        eff_dur = w_end - w_start - (j_count * J_DURATION)
        if eff_dur <= 0: continue
        v_part = eff_dur / len(vowels_only)
        
        curr_t = w_start
        for p in phonemes:
            if p == 'й':
                curr_t += J_DURATION
                continue
            
            v_start, v_end = curr_t, curr_t + v_part
            f1, f2, dur, f0, inten, jit, shim, hnr = find_acoustic_features(sound, formant_obj, pitch_obj, intensity_obj, v_start, v_end)
            
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f0):
                # НОВАЯ ФОРМУЛА ЭНЕРГИИ (обновленная)
                impulses = f0 * dur
                total_energy = 0.00012 * impulses - 0.00015
                
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': f1, 'F2': f2, 'duration': dur,
                    'mean_pitch': f0, 'mean_intensity': inten, 'total_energy': total_energy,
                    'jitter': jit, 'shimmer': shim, 'hnr': hnr
                })
            curr_t = v_end
            
    return vowel_data

# --- Визуализация (3D Карта 1 в 1) ---
def plot_3d_vowel_count(vowel_data, audio_filename):
    if not vowel_data: return None, None
    
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    df = pd.DataFrame(vowel_data)
    
    # Агрегация данных для 3D
    plot_data_dict = {}
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if not v_df.empty:
            plot_data_dict[v] = {
                'avg_F1': v_df['F1'].mean(),
                'avg_F2': v_df['F2'].mean(),
                'avg_intensity': v_df['mean_intensity'].mean(),
                'avg_energy': v_df['total_energy'].mean(),
                'count': len(v_df),
                'avg_pulses': (v_df['mean_pitch'] * v_df['duration']).mean()
            }

    if not plot_data_dict: return None, None

    x_coords, y_coords, z_heights, vowel_labels, marker_sizes, hover_texts = [], [], [], [], [], []
    
    # Нормализация размеров маркеров (по интенсивности)
    ints = [d['avg_intensity'] for d in plot_data_dict.values()]
    min_i, max_i = min(ints), max(ints)

    for v in vowel_order:
        if v in plot_data_dict:
            d = plot_data_dict[v]
            x_coords.append(d['avg_F1'])
            y_coords.append(d['avg_F2'])
            z_heights.append(d['count'])
            vowel_labels.append(v)
            
            m_size = 10 + (d['avg_intensity'] - min_i)/(max_i - min_i + 1e-6) * 30
            marker_sizes.append(m_size)
            
            hover_texts.append(
                f"Фонема: {v}<br>Количество: {d['count']}<br>F1: {d['avg_F1']:.0f} Гц<br>"
                f"F2: {d['avg_F2']:.0f} Гц<br>Энергия: {d['avg_energy']:.6f}"
            )

    # Замыкание контура и-ы-у-о-а-э-и
    if x_coords:
        x_conn, y_conn, z_conn = x_coords + [x_coords[0]], y_coords + [y_coords[0]], z_heights + [z_heights[0]]
    
    fig = go.Figure()

    # Вертикальные линии (столбики)
    for i in range(len(x_coords)):
        fig.add_trace(go.Scatter3d(
            x=[x_coords[i], x_coords[i]], y=[y_coords[i], y_coords[i]], z=[0, z_heights[i]],
            mode='lines', line=dict(color='gray', width=5), showlegend=False
        ))

    # Точки основания
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=[0]*len(z_heights),
        mode='markers+text', text=vowel_labels, textposition="bottom center",
        marker=dict(size=marker_sizes, color=np.arange(len(x_coords)), colorscale='Viridis'),
        hovertext=hover_texts, name='База гласных'
    ))

    # Красная линия по вершинам (и-ы-у-о-а-э-и)
    fig.add_trace(go.Scatter3d(
        x=x_conn, y=y_conn, z=z_conn,
        mode='lines+markers', line=dict(color='red', width=6),
        marker=dict(size=4, color='red'), name='Цепочка и-ы-у-о-а-э-и'
    ))

    fig.update_layout(
        title=f"3D Карта Гласных: {os.path.basename(audio_filename)}",
        scene=dict(
            xaxis_title='F1 (Гц)', yaxis_title='F2 (Гц)', zaxis_title='Количество',
            xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")
        ),
        width=1000, height=800
    )
    return fig, plot_data_dict

# --- Прочие графики (Звезда, Кластеры) ---
def get_russian_norms(gender='female'):
    if gender == "мужчина":
        return {'и':{'F1':290,'F2':2150,'dur':0.075,'F0':125},'ы':{'F1':420,'F2':1350,'dur':0.080,'F0':120},
                'у':{'F1':320,'F2':820,'dur':0.088,'F0':115},'о':{'F1':460,'F2':920,'dur':0.092,'F0':118},
                'а':{'F1':690,'F2':1300,'dur':0.108,'F0':115},'э':{'F1':490,'F2':1750,'dur':0.082,'F0':122}}
    return {'и':{'F1':320,'F2':2250,'dur':0.078,'F0':215},'ы':{'F1':450,'F2':1400,'dur':0.082,'F0':205},
            'у':{'F1':340,'F2':850,'dur':0.090,'F0':195},'о':{'F1':480,'F2':950,'dur':0.095,'F0':200},
            'а':{'F1':720,'F2':1350,'dur':0.110,'F0':195},'э':{'F1':520,'F2':1850,'dur':0.085,'F0':210}}

def plot_radar_vowel_star(vowel_data, gender='female'):
    norms = get_russian_norms(gender)
    df = pd.DataFrame(vowel_data)
    agg = df.groupby('vowel').mean().reindex(['и', 'ы', 'у', 'о', 'а', 'э'])
    
    fig = go.Figure()
    for v in agg.index:
        if v not in norms or pd.isna(agg.loc[v, 'F1']): continue
        p, n = agg.loc[v], norms[v]
        vals = [(p['F1']-n['F1'])/n['F1']*100, (p['F2']-n['F2'])/n['F2']*100, (p['duration']-n['dur'])/n['dur']*100, 12*np.log2(p['mean_pitch']/n['F0'])]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=['F1%','F2%','Dur%','Pitch','F1%'], name=v, fill='toself'))
    
    fig.update_layout(polar=dict(radialaxis=dict(range=[-50, 50])), title="Радиальная звезда (отклонения от нормы)")
    return fig

# --- Main ---
def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D Pro")
    st.title("SpeechViz3D: 3D Карта, Микропараметры и Звезда Гласных")

    uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])

    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f: f.write(uploaded_file.getbuffer())

        # Проверка кэша файла
        if "vowel_data" not in st.session_state or st.session_state.get("file") != uploaded_file.name:
            segments = transcribe_cached(audio_path)
            vowel_data = analyze_vowels_cached(audio_path, segments)
            st.session_state.vowel_data = vowel_data
            st.session_state.file = uploaded_file.name

        vowel_data = st.session_state.vowel_data
        df = pd.DataFrame(vowel_data)

        # 1. Сводка и микропараметры
        st.subheader("Акустические параметры (Средние значения)")
        summary_df = df.groupby('vowel').agg({
            'F1': 'mean', 'F2': 'mean', 'total_energy': 'mean',
            'jitter': 'mean', 'shimmer': 'mean', 'hnr': 'mean'
        }).round(5)
        st.table(summary_df)

        # 2. 3D Карта (как в запросе)
        st.subheader("3D Карта Гласных (Контур и-ы-у-о-а-э-и)")
        fig_3d, _ = plot_3d_vowel_count(vowel_data, uploaded_file.name)
        if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Звезда отклонений")
            gender = st.radio("Пол", ["женщина", "мужчина"])
            st.plotly_chart(plot_radar_vowel_star(vowel_data, gender))
        with col2:
            st.subheader("Распределение")
            st.plotly_chart(px.histogram(df, x="vowel", color="vowel", title="Количество фонем"))

        # Скачивание CSV
        st.download_button("Скачать полные данные (CSV)", df.to_csv(index=False).encode('utf-8-sig'), "speech_data.csv", "text/csv")

if __name__ == "__main__":
    main()
