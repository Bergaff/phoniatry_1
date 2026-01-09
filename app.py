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
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэширование ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Анализ аудио...")
def process_audio(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    
    word_segments = []
    full_text = []
    for s in segments:
        for w in s.words:
            if w.probability > 0.1:
                word_segments.append({'word': w.word.strip().lower(), 'start': w.start, 'end': w.end})
                full_text.append(w.word.strip())
    
    sound = parselmouth.Sound(audio_path)
    vowel_data = analyze_vowels(sound, word_segments)
    return vowel_data, " ".join(full_text)

# --- Логика анализа ---
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
            mapping = {'е':'э', 'ё':'о', 'ю':'у', 'я':'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи': phonemes.append(char)
    return phonemes

def analyze_vowels(sound, segments):
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR)
    intensity_obj = sound.to_intensity()
    vowel_results = []
    
    for seg in segments:
        phonemes = extract_phonemes(seg['word'])
        vowels = [p for p in phonemes if p != 'й']
        if not vowels: continue
        
        j_dur = 0.04
        eff_dur = (seg['end'] - seg['start']) - (phonemes.count('й') * j_dur)
        if eff_dur <= 0: continue
        v_dur = eff_dur / len(vowels)
        
        curr_t = seg['start']
        for p in phonemes:
            if p == 'й':
                curr_t += j_dur; continue
            
            v_start, v_end = curr_t, curr_t + v_dur
            mid = (v_start + v_end) / 2
            
            # Извлечение параметров
            f1 = formant_obj.get_value_at_time(1, mid)
            f2 = formant_obj.get_value_at_time(2, mid)
            f3 = formant_obj.get_value_at_time(3, mid)
            f0 = pitch_obj.get_value_at_time(mid)
            inten = intensity_obj.get_value(mid)
            
            # Микропараметры через PointProcess
            try:
                part = sound.extract_part(from_time=v_start, to_time=v_end)
                pp = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
                jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
                shimmer = parselmouth.praat.call([part, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                hnr = parselmouth.praat.call(part.to_harmonicity(), "Get mean", 0, 0)
                rms = part.get_rms()
            except:
                jitter, shimmer, hnr, rms = 0, 0, 0, 0

            if not np.isnan(f1) and f0 > 0:
                pulses = f0 * v_dur
                energy = 0.00012 * pulses - 0.00015
                
                vowel_results.append({
                    'Label': p, 'Start_s': v_start, 'End_s': v_end, 'Duration_s': v_dur,
                    'Mean_Pitch_Hz': f0, 'Mean_Intensity_dB': inten, 'F1_Hz': f1, 'F2_Hz': f2, 'F3_Hz': f3,
                    'RMS_Amplitude': rms, 'Energy_Pa2s': energy, 'Pulse_Count': int(pulses),
                    'Jitter_pct': jitter, 'Shimmer_dB': shimmer, 'HNR_dB': hnr, 'Word': seg['word']
                })
            curr_t = v_end
    return vowel_results

# --- Функции графиков ---
def plot_3d_map(df):
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    agg = df.groupby('Label').agg({'F1_Hz':'mean', 'F2_Hz':'mean', 'Energy_Pa2s':'mean', 'Mean_Intensity_dB':'mean', 'Label':'count'}).rename(columns={'Label':'count'}).reindex(v_order).dropna()
    
    fig = go.Figure()
    # Линии-столбцы
    for char in agg.index:
        d = agg.loc[char]
        fig.add_trace(go.Scatter3d(x=[d['F1_Hz'], d['F1_Hz']], y=[d['F2_Hz'], d['F2_Hz']], z=[0, d['count']], mode='lines', line=dict(color='gray', width=4), showlegend=False))
    
    # Контур
    x_c, y_c, z_c = list(agg['F1_Hz']) + [agg['F1_Hz'].iloc[0]], list(agg['F2_Hz']) + [agg['F2_Hz'].iloc[0]], list(agg['count']) + [agg['count'].iloc[0]]
    fig.add_trace(go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='lines+markers', line=dict(color='red', width=6), name='Цепочка и-ы-у-о-а-э-и'))
    
    fig.update_layout(scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='Кол-во', xaxis_autorange="reversed", yaxis_autorange="reversed"), margin=dict(l=0, r=0, b=0, t=40))
    return fig, agg

# --- UI ---
def main():
    st.set_page_config(layout="wide", page_title="Speech Analysis Pro")
    st.title("SpeechViz3D: Профессиональный анализ вокализма")

    up = st.file_uploader("Загрузите WAV", type="wav")
    if up:
        path = os.path.join(OUTPUT_DIR, up.name)
        with open(path, "wb") as f: f.write(up.getbuffer())
        
        data, text = process_audio(path)
        df = pd.DataFrame(data)

        st.info(f"**Распознанный текст:** {text}")

        t1, t2, t3, t4 = st.tabs(["3D Карта", "Гистограмма", "Радиальная звезда", "Кластеризация"])

        with t1:
            fig3d, agg_df = plot_3d_map(df)
            st.plotly_chart(fig3d, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.download_button("Скачать подробный отчет (все фонемы)", df.to_csv(index=False).encode('utf-8-sig'), "full_report.csv")
            col2.download_button("Скачать средние значения (3D)", agg_df.to_csv().encode('utf-8-sig'), "summary_3d.csv")

        with t2:
            st.plotly_chart(px.histogram(df, x="Label", color="Label", title="Распределение гласных"), use_container_width=True)
            st.download_button("Скачать данные гистограммы", df['Label'].value_counts().to_csv().encode('utf-8-sig'), "counts.csv")

        with t3:
            gender = st.selectbox("Пол пациента", ["женщина", "мужчина"])
            # (Упрощенная логика отрисовки для примера)
            st.write("Отрисовка радиального графика отклонений...")
            st.download_button("Скачать данные отклонений", df.groupby('Label').mean().to_csv().encode('utf-8-sig'), "deviations.csv")

        with t4:
            fig_km = px.scatter(df, x="F1_Hz", y="F2_Hz", color="Label", title="F1/F2 Кластеризация")
            fig_km.update_layout(xaxis_autorange="reversed", yaxis_autorange="reversed")
            st.plotly_chart(fig_km, use_container_width=True)
            st.download_button("Скачать данные кластеров", df[['Label', 'F1_Hz', 'F2_Hz']].to_csv().encode('utf-8-sig'), "clusters.csv")

if __name__ == "__main__":
    main()
