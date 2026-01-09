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

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL_NAME = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэширование ресурсов и моделей ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL_NAME, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация аудио...")
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

@st.cache_data(show_spinner="Акустический анализ (Praat)...")
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

def find_acoustic_features(formant_obj, pitch_obj, intensity_obj, start, end):
    F1_v, F2_v, F0_v, Int_v = [], [], [], []
    t = start
    while t < end:
        f1 = formant_obj.get_value_at_time(1, t)
        f2 = formant_obj.get_value_at_time(2, t)
        f0 = pitch_obj.get_value_at_time(t)
        intensity = intensity_obj.get_value(t)
        if not math.isnan(f1): F1_v.append(f1)
        if not math.isnan(f2): F2_v.append(f2)
        if not math.isnan(f0): F0_v.append(f0)
        if not math.isnan(intensity): Int_v.append(intensity)
        t += 0.005
    return (np.nanmedian(F1_v) if F1_v else np.nan, 
            np.nanmedian(F2_v) if F2_v else np.nan, 
            end - start, 
            np.nanmedian(F0_v) if F0_v else np.nan, 
            np.nanmedian(Int_v) if Int_v else np.nan)

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

    for segment in transcription_segments:
        word, w_start, w_end = segment['word'], segment['start'], segment['end']
        phonemes = extract_phonemes(word)
        if not phonemes: continue
        
        v_count = len([p for p in phonemes if p != 'й'])
        j_count = phonemes.count('й')
        eff_dur = w_end - w_start - (j_count * J_DURATION)
        if v_count == 0 or eff_dur <= 0: continue
        
        v_dur = eff_dur / v_count
        curr_t = w_start
        for p in phonemes:
            if p == 'й':
                curr_t += J_DURATION
                continue
            f1, f2, dur, f0, inten = find_acoustic_features(formant_obj, pitch_obj, intensity_obj, curr_t, curr_t + v_dur)
            if not np.isnan(f1) and not np.isnan(f2):
                energy = 0.00012 * (f0 * dur) - 0.00015
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': f1, 'F2': f2, 
                    'duration': dur, 'mean_pitch': f0, 'mean_intensity': inten, 'total_energy': max(energy, 0)
                })
            curr_t += v_dur
    return vowel_data

def normalize_lobanov(df):
    for col in ['F1', 'F2']:
        mean_v, std_v = df[col].mean(), df[col].std()
        df[f'{col}_z'] = (df[col] - mean_v) / (std_v if std_v != 0 else 1)
    return df

def get_russian_norms(gender):
    if gender == "мужчина":
        return {
            'и': {'F1': 290, 'F2': 2150, 'duration': 0.075, 'F0': 125},
            'ы': {'F1': 420, 'F2': 1350, 'duration': 0.080, 'F0': 120},
            'у': {'F1': 320, 'F2': 820,  'duration': 0.088, 'F0': 115},
            'о': {'F1': 460, 'F2': 920,  'duration': 0.092, 'F0': 118},
            'а': {'F1': 690, 'F2': 1300, 'duration': 0.108, 'F0': 115},
            'э': {'F1': 490, 'F2': 1750, 'duration': 0.082, 'F0': 122},
        }
    return {
        'и': {'F1': 320, 'F2': 2250, 'duration': 0.078, 'F0': 215},
        'ы': {'F1': 450, 'F2': 1400, 'duration': 0.082, 'F0': 205},
        'у': {'F1': 340, 'F2': 850,  'duration': 0.090, 'F0': 195},
        'о': {'F1': 480, 'F2': 950,  'duration': 0.095, 'F0': 200},
        'а': {'F1': 720, 'F2': 1350, 'duration': 0.110, 'F0': 195},
        'э': {'F1': 520, 'F2': 1850, 'duration': 0.085, 'F0': 210},
    }

# --- ФУНКЦИИ ГРАФИКОВ ---
def plot_3d_vowel_count(vowel_data, base_name):
    df = pd.DataFrame(vowel_data)
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    # Исправлено: numeric_only=True для новых версий Pandas
    agg = df.groupby('vowel').mean(numeric_only=True).reindex(vowel_order)
    counts = df['vowel'].value_counts().reindex(vowel_order).fillna(0)
    
    fig = go.Figure()
    x, y, z = agg['F1'].tolist(), agg['F2'].tolist(), counts.tolist()
    
    # Добавляем замыкающую точку для линии
    x_conn, y_conn, z_conn = x + [x[0]], y + [y[0]], z + [z[0]]
    
    fig.add_trace(go.Scatter3d(x=x_conn, y=y_conn, z=z_conn, mode='lines+markers', line=dict(color='red', width=5)))
    
    for i, v in enumerate(vowel_order):
        if not np.isnan(x[i]):
            fig.add_trace(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[0, z[i]], mode='lines', line=dict(color='gray')))

    fig.update_layout(scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='Кол-во',
                                 xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")),
                      title=f"3D Карта: {base_name}", width=900, height=700)
    return fig

def plot_radar_vowel_star(vowel_data, gender):
    df = pd.DataFrame(vowel_data)
    norms = get_russian_norms(gender)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    # Исправлено: numeric_only=True
    agg = df.groupby('vowel').mean(numeric_only=True).reindex(v_order)
    
    fig = go.Figure()
    for v in v_order:
        if v in agg.index and not np.isnan(agg.loc[v, 'F1']):
            p, n = agg.loc[v], norms[v]
            vals = [
                (p['F1']-n['F1'])/n['F1']*100, (p['F2']-n['F2'])/n['F2']*100,
                (p['duration']-n['duration'])/n['duration']*100, 
                12 * np.log2(p['mean_pitch']/n['F0'])
            ]
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=['F1 %','F2 %','Длит. %','Тон','F1 %'], 
                                          fill='toself', name=v))
    
    fig.update_layout(polar=dict(radialaxis=dict(range=[-100, 100])), title=f"Звезда ({gender})")
    return fig, agg

@st.cache_data(show_spinner="Кластеризация...")
def get_kmeans_plot(vowel_data):
    df = pd.DataFrame(vowel_data)
    if len(df) < 6: return None
    df_n = normalize_lobanov(df.copy())
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df_n['cluster'] = kmeans.fit_predict(df_n[['F1_z', 'F2_z']])
    
    fig = px.scatter(df_n, x='F1', y='F2', color='cluster', text='vowel', title="K-means кластеризация")
    fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))
    return fig

# --- MAIN ---
def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D")
    st.title("Анализ гласных фонем")
    
    up = st.file_uploader("WAV файл", type=["wav"])
    if up:
        path = os.path.join(OUTPUT_DIR, up.name)
        with open(path, "wb") as f: f.write(up.getbuffer())
        
        # Анализ
        segments = transcribe_cached(path)
        v_data = analyze_vowels_cached(path, segments)
        
        if v_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("3D Визуализация")
                st.plotly_chart(plot_3d_vowel_count(v_data, up.name), use_container_width=True)
            
            with col2:
                st.subheader("Кластеризация")
                km_fig = get_kmeans_plot(v_data)
                if km_fig: st.plotly_chart(km_fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Радиальный анализ")
            # Выбор пола сделан вне кэша, чтобы переключалось быстро
            gender = st.radio("Пол для сравнения с нормой:", ["женщина", "мужчина"], horizontal=True)
            radar_fig, radar_df = plot_radar_vowel_star(v_data, gender)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.dataframe(pd.DataFrame(v_data).head(10))
        else:
            st.error("Гласные не найдены. Попробуйте другую запись.")

if __name__ == "__main__":
    main()
