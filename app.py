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
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэширование ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Обработка аудио...")
def process_audio_full(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    
    word_segments = []
    text_pieces = []
    for s in segments:
        for w in s.words:
            if w.probability > 0.1:
                word_segments.append({'word': w.word.strip().lower(), 'start': w.start, 'end': w.end})
                text_pieces.append(w.word.strip())
    
    sound = parselmouth.Sound(audio_path)
    vowel_data = analyze_vowels_complex(sound, word_segments)
    return vowel_data, " ".join(text_pieces)

# --- Логика анализа ---
def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя' and (i == 0 or text_clean[i-1] not in 'аоуэыиьъ'):
            mapping = {'е':['й','э'], 'ё':['й','о'], 'ю':['й','у'], 'я':['й','а']}
            phonemes.extend(mapping[char])
        elif char in 'еёюя':
            mapping = {'е':'э', 'ё':'о', 'ю':'у', 'я':'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def analyze_vowels_complex(sound, segments):
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR)
    intensity_obj = sound.to_intensity()
    results = []
    
    for seg in segments:
        phonemes = extract_phonemes(seg['word'])
        vowels_only = [p for p in phonemes if p != 'й']
        if not vowels_only: continue
        
        j_dur = 0.04
        eff_dur = (seg['end'] - seg['start']) - (phonemes.count('й') * j_dur)
        if eff_dur <= 0: continue
        v_dur = eff_dur / len(vowels_only)
        
        curr_t = seg['start']
        for p in phonemes:
            if p == 'й':
                curr_t += j_dur
                continue
            
            v_start, v_end = curr_t, curr_t + v_dur
            mid = (v_start + v_end) / 2
            
            f1 = formant_obj.get_value_at_time(1, mid)
            f2 = formant_obj.get_value_at_time(2, mid)
            f3 = formant_obj.get_value_at_time(3, mid)
            f0 = pitch_obj.get_value_at_time(mid)
            inten = intensity_obj.get_value(mid)
            
            if not np.isnan(f1) and f0 > 0:
                # Новая формула энергии и импульсов
                pulses = f0 * v_dur
                energy = 0.00012 * pulses - 0.00015
                
                # Микропараметры
                try:
                    part = sound.extract_part(from_time=v_start, to_time=v_end)
                    pp = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
                    jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
                    shimmer = parselmouth.praat.call([part, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    hnr = parselmouth.praat.call(part.to_harmonicity(), "Get mean", 0, 0)
                    rms = part.get_rms()
                except:
                    jitter, shimmer, hnr, rms = 0, 0, 0, 0

                results.append({
                    'Label': p, 'Start_s': v_start, 'End_s': v_end, 'Duration_s': v_dur,
                    'Mean_Pitch_Hz': f0, 'Mean_Intensity_dB': inten, 'F1_Hz': f1, 'F2_Hz': f2, 'F3_Hz': f3,
                    'RMS_Amplitude': rms, 'Energy_Pa2s': energy, 'Pulse_Count': int(pulses),
                    'Jitter_pct': jitter, 'Shimmer_dB': shimmer, 'HNR_dB': hnr, 'Word': seg['word']
                })
            curr_t = v_end
    return results

# --- Визуализация ---
def plot_3d_vowel_map(df):
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    # Исправленная агрегация (numeric_only=True решает ошибку в pandas)
    agg = df.groupby('Label').mean(numeric_only=True).reindex(v_order).dropna(subset=['F1_Hz'])
    counts = df['Label'].value_counts().reindex(v_order).fillna(0)
    agg['count'] = counts

    fig = go.Figure()

    # Линии-столбцы от пола до пика
    for char in agg.index:
        d = agg.loc[char]
        fig.add_trace(go.Scatter3d(
            x=[d['F1_Hz'], d['F1_Hz']], y=[d['F2_Hz'], d['F2_Hz']], z=[0, d['count']],
            mode='lines', line=dict(color='gray', width=5), showlegend=False
        ))

    # Красная цепочка и-ы-у-о-а-э-и
    x_c = list(agg['F1_Hz']) + [agg['F1_Hz'].iloc[0]]
    y_c = list(agg['F2_Hz']) + [agg['F2_Hz'].iloc[0]]
    z_c = list(agg['count']) + [agg['count'].iloc[0]]
    
    fig.add_trace(go.Scatter3d(
        x=x_c, y=y_c, z=z_c, mode='lines+markers',
        line=dict(color='red', width=6), marker=dict(size=5, color='red'),
        name='Цепочка гласных'
    ))

    # Базовые точки (на полу)
    fig.add_trace(go.Scatter3d(
        x=agg['F1_Hz'], y=agg['F2_Hz'], z=[0]*len(agg),
        mode='markers+text', text=agg.index,
        marker=dict(size=10, color=np.arange(len(agg)), colorscale='Viridis'),
        name='Фонемы'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='F1 (Гц)', autorange="reversed"),
            yaxis=dict(title='F2 (Гц)', autorange="reversed"),
            zaxis=dict(title='Количество')
        ),
        width=1000, height=800
    )
    return fig, agg

# --- Главное приложение ---
def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D Pro")
    st.title("SpeechViz3D: Профессиональный акустический анализ")

    up = st.file_uploader("Загрузите аудио (WAV)", type="wav")
    
    if up:
        path = os.path.join(OUTPUT_DIR, up.name)
        with open(path, "wb") as f: f.write(up.getbuffer())
        
        vowel_results, full_text = process_audio_full(path)
        df = pd.DataFrame(vowel_results)

        st.success(f"**Распознанный текст:** {full_text}")

        tabs = st.tabs(["3D Карта гласных", "Гистограмма", "Радиальная звезда", "Кластеризация"])

        with tabs[0]:
            fig3d, agg_df = plot_3d_vowel_map(df)
            st.plotly_chart(fig3d, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.download_button("Скачать подробную таблицу (Label, F1-F3, Jitter...)", 
                               df.to_csv(index=False).encode('utf-8-sig'), "full_vowel_data.csv", "text/csv")
            # Исправленная выгрузка средних
            c2.download_button("Скачать таблицу средних значений", 
                               agg_df.to_csv().encode('utf-8-sig'), "average_vowels.csv", "text/csv")

        with tabs[1]:
            st.plotly_chart(px.histogram(df, x="Label", color="Label", title="Частота появления гласных"))
            st.download_button("Скачать данные гистограммы", df['Label'].value_counts().to_csv().encode('utf-8-sig'), "counts.csv")

        with tabs[2]:
            gender = st.selectbox("Пол пациента", ["женщина", "мужчина"])
            st.write("Визуализация радиального отклонения от нормы (в разработке)...")
            # Исправленная ошибка groupby().mean()
            radar_data = df.groupby('Label').mean(numeric_only=True)
            st.download_button("Скачать данные для звезды", radar_data.to_csv().encode('utf-8-sig'), "deviations.csv")

        with tabs[3]:
            fig_km = px.scatter(df, x="F1_Hz", y="F2_Hz", color="Label", title="F1/F2 Кластеры")
            fig_km.update_layout(xaxis_autorange="reversed", yaxis_autorange="reversed")
            st.plotly_chart(fig_km, use_container_width=True)
            st.download_button("Скачать данные кластеризации", df[['Label','F1_Hz','F2_Hz']].to_csv().encode('utf-8-sig'), "clusters.csv")

if __name__ == "__main__":
    main()
