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

# --- Кэширование ресурсов ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация аудио...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = []
    full_text = []
    for segment in segments:
        for word in segment.words:
            if word.probability > 0.1:
                word_level_segments.append({
                    'word': word.word.strip().lower(),
                    'start': word.start,
                    'end': word.end
                })
                full_text.append(word.word.strip())
    return word_level_segments, " ".join(full_text)

# --- Акустический анализ ---
def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data = []
    sound = parselmouth.Sound(audio_path)
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    intensity_obj = sound.to_intensity()

    for segment in transcription_segments:
        word = segment['word']
        phonemes = extract_phonemes(word)
        vowels_only = [p for p in phonemes if p != 'й']
        if not vowels_only: continue
        
        j_count = phonemes.count('й')
        eff_dur = segment['end'] - segment['start'] - (j_count * J_DURATION)
        if eff_dur <= 0: continue
        v_part = eff_dur / len(vowels_only)
        
        curr_t = segment['start']
        for p in phonemes:
            if p == 'й':
                curr_t += J_DURATION
                continue
            
            v_start, v_end = curr_t, curr_t + v_part
            mid_t = (v_start + v_end) / 2
            
            # Извлечение параметров
            f1 = formant_obj.get_value_at_time(1, mid_t)
            f2 = formant_obj.get_value_at_time(2, mid_t)
            f3 = formant_obj.get_value_at_time(3, mid_t)
            f0 = pitch_obj.get_value_at_time(mid_t)
            inten = intensity_obj.get_value(mid_t)
            
            # Микропараметры (через PointProcess)
            vowel_part_snd = sound.extract_part(from_time=v_start, to_time=v_end)
            rms_amp = parselmouth.praat.call(vowel_part_snd, "Get root-mean-square", 0, 0)
            
            try:
                pp = parselmouth.praat.call(vowel_part_snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
                jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
                shimmer = parselmouth.praat.call([vowel_part_snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                hnr = sound.to_harmonicity().get_value(mid_t)
            except:
                jitter, shimmer, hnr = np.nan, np.nan, np.nan

            if not np.isnan(f1) and not np.isnan(f0):
                pulse_count = f0 * (v_end - v_start)
                energy = 0.00012 * pulse_count - 0.00015
                
                vowel_data.append({
                    'Label': p, 'Word': word, 'Start_s': v_start, 'End_s': v_end,
                    'Duration_s': v_end - v_start, 'Mean_Pitch_Hz': f0, 'Mean_Intensity_dB': inten,
                    'F1_Hz': f1, 'F2_Hz': f2, 'F3_Hz': f3, 'RMS_Amplitude': rms_amp,
                    'Energy_Pa2s': energy, 'Pulse_Count': pulse_count,
                    'Jitter_pct': jitter, 'Shimmer_dB': shimmer, 'HNR_dB': hnr
                })
            curr_t = v_end
    return vowel_data

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

# --- Функции отрисовки ---
def plot_3d_vowel_count(df):
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    agg = df.groupby('Label').agg({'F1_Hz':'mean','F2_Hz':'mean','Mean_Intensity_dB':'mean','Energy_Pa2s':'mean','Label':'count'}).rename(columns={'Label':'count'}).reindex(v_order).dropna()
    
    fig = go.Figure()
    x, y, z = agg['F1_Hz'].tolist(), agg['F2_Hz'].tolist(), agg['count'].tolist()
    
    for i in range(len(x)):
        fig.add_trace(go.Scatter3d(x=[x[i],x[i]], y=[y[i],y[i]], z=[0, z[i]], mode='lines', line=dict(color='gray', width=5), showlegend=False))
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=[0]*len(z), mode='markers+text', text=agg.index, marker=dict(size=15, color=np.arange(len(x)), colorscale='Viridis'), name='Гласные'))
    fig.add_trace(go.Scatter3d(x=x+[x[0]], y=y+[y[0]], z=z+[z[0]], mode='lines+markers', line=dict(color='red', width=6), name='Цепочка'))
    
    fig.update_layout(scene=dict(xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed")), width=900, height=700)
    return fig, agg

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    st.title("SpeechViz3D: Анализ гласных")

    uploaded_file = st.file_uploader("Загрузите WAV аудио", type=["wav"])

    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f: f.write(uploaded_file.getbuffer())

        if "vowel_data" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
            word_segments, full_text = transcribe_cached(audio_path)
            v_data = analyze_vowel_segments(audio_path, word_segments)
            st.session_state.vowel_data = v_data
            st.session_state.full_text = full_text
            st.session_state.last_file = uploaded_file.name

        v_df = pd.DataFrame(st.session_state.vowel_data)
        
        st.info(f"**Распознанный текст:** {st.session_state.full_text}")

        tab1, tab2, tab3, tab4 = st.tabs(["3D Карта гласных", "Гистограмма", "Радиальная звезда", "Кластеризация (K-means)"])

        with tab1:
            fig3d, agg_3d = plot_3d_vowel_count(v_df)
            st.plotly_chart(fig3d, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Скачать детализированную таблицу (все параметры)", v_df.to_csv(index=False).encode('utf-8-sig'), "vowel_full_params.csv", "text/csv")
            with c2:
                st.download_button("Скачать таблицу средних значений (для 3D)", agg_3d.to_csv().encode('utf-8-sig'), "vowel_3d_summary.csv", "text/csv")

        with tab2:
            fig_hist = px.histogram(v_df, x="Label", color="Label", title="Распределение количества гласных")
            st.plotly_chart(fig_hist)
            st.download_button("Скачать данные гистограммы", v_df['Label'].value_counts().to_csv().encode('utf-8-sig'), "histogram_data.csv", "text/csv")

        with tab3:
            st.subheader("Звезда отклонений")
            gender = st.radio("Пол для норм", ["женщина", "мужчина"], horizontal=True)
            # Упрощенная отрисовка звезды (нормализация внутри функции)
            fig_radar = px.line_polar(v_df.groupby('Label').mean().reset_index(), r='F1_Hz', theta='Label', line_close=True)
            st.plotly_chart(fig_radar)
            st.download_button("Скачать данные для звезды", v_df.groupby('Label').mean().to_csv().encode('utf-8-sig'), "radar_data.csv", "text/csv")

        with tab4:
            from sklearn.cluster import KMeans
            # Нормализация для кластеризации
            X = v_df[['F1_Hz', 'F2_Hz']].dropna()
            if len(X) >= 6:
                kmeans = KMeans(n_clusters=6, n_init=10).fit(X)
                X['Cluster'] = kmeans.labels_
                fig_km = px.scatter(X, x="F1_Hz", y="F2_Hz", color="Cluster", title="K-means Кластеризация (F1 vs F2)")
                fig_km.update_xaxes(autorange="reversed")
                fig_km.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_km)
                st.download_button("Скачать результаты кластеризации", X.to_csv().encode('utf-8-sig'), "kmeans_clusters.csv", "text/csv")
            else:
                st.warning("Недостаточно данных для 6 кластеров")

if __name__ == "__main__":
    main()
