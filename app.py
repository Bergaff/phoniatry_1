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

# --- Вспомогательные функции ---
def extract_phonemes(text):
    phonemes = []
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        if char in 'еёюя' and (i == 0 or text_clean[i-1] not in 'аоуэыиьъ'):
            mapping = {'е': ['й','э'], 'ё': ['й','о'], 'ю': ['й','у'], 'я': ['й','а']}
            phonemes.extend(mapping[char])
        elif char in 'еёюя':
            mapping = {'е': 'э', 'ё': 'о', 'ю': 'у', 'я': 'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи':
            phonemes.append(char)
    return phonemes

def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
        formant_obj = sound.to_formant_burg()
        pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR)
        intensity_obj = sound.to_intensity()
        
        for segment in transcription_segments:
            word, word_start, word_end = segment['word'], segment['start'], segment['end']
            phonemes_in_word = extract_phonemes(word)
            if not phonemes_in_word: continue
            
            vowels_only = [p for p in phonemes_in_word if p != 'й']
            if not vowels_only: continue
            
            v_dur = (word_end - word_start - (phonemes_in_word.count('й') * J_DURATION)) / len(vowels_only)
            current_time = word_start
            
            for ph in phonemes_in_word:
                if ph == 'й':
                    current_time += J_DURATION
                    continue
                
                mid_t = current_time + v_dur / 2
                f1 = formant_obj.get_value_at_time(1, mid_t)
                f2 = formant_obj.get_value_at_time(2, mid_t)
                f0 = pitch_obj.get_value_at_time(mid_t)
                intn = intensity_obj.get_value(mid_t)
                
                if not (math.isnan(f1) or math.isnan(f2)):
                    vowel_data.append({
                        'vowel': ph, 'F1': f1, 'F2': f2, 'duration': v_dur,
                        'mean_pitch': f0 if not math.isnan(f0) else 120,
                        'mean_intensity': intn, 'total_energy': (intn * v_dur) / 100
                    })
                current_time += v_dur
    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
    return vowel_data

def get_russian_norms(gender):
    if gender == "мужчина":
        return {'и':{'F1':290,'F2':2150}, 'ы':{'F1':420,'F2':1350}, 'у':{'F1':320,'F2':820}, 
                'о':{'F1':460,'F2':920}, 'а':{'F1':690,'F2':1300}, 'э':{'F1':490,'F2':1750}}
    return {'и':{'F1':320,'F2':2250}, 'ы':{'F1':450,'F2':1400}, 'у':{'F1':340,'F2':850}, 
            'о':{'F1':480,'F2':950}, 'а':{'F1':720,'F2':1350}, 'э':{'F1':520,'F2':1850}}

# --- Визуализация ---
def plot_3d_vowels_custom(vowel_data):
    df = pd.DataFrame(vowel_data)
    vowel_colors = {'а':'red', 'о':'orange', 'у':'purple', 'ы':'pink', 'и':'blue', 'э':'green'}
    
    fig = go.Figure()
    
    # 1. Линии от пола до точек
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['F1'], row['F1']], y=[row['F2'], row['F2']], z=[0, row['total_energy']],
            mode='lines', line=dict(color='lightgray', width=2), showlegend=False, hoverinfo='skip'
        ))

    # 2. Сами шары (Размер = Интенсивность, Цвет = Фонема)
    for v, group in df.groupby('vowel'):
        fig.add_trace(go.Scatter3d(
            x=group['F1'], y=group['F2'], z=group['total_energy'],
            mode='markers',
            name=f"Фонема: {v}",
            marker=dict(
                size=group['mean_intensity'] / 2, # Размер шара от интенсивности
                color=vowel_colors.get(v, 'gray'),
                opacity=0.8,
                symbol='sphere'
            ),
            text=group['vowel'],
            hovertemplate="Фонема: %{text}<br>F1: %{x:.0f}<br>F2: %{y:.0f}<br>Энергия: %{z:.4f}"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='F1 (Гц)', autorange="reversed"),
            yaxis=dict(title='F2 (Гц)', autorange="reversed"),
            zaxis=dict(title='Энергия / Высота')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Карта Гласных (Размер шара = Громкость)"
    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="SpeechViz 3D")
    st.title("🎤 Анализ артикуляции: 3D визуализация")

    uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])

    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        segments = transcribe_cached(audio_path)
        vowel_data = analyze_vowel_segments(audio_path, segments)

        if vowel_data:
            # Создаем вкладки
            tab1, tab2, tab3 = st.tabs(["📊 3D Карта", "📈 Статистика", "⭐ Звезда Норм"])

            with tab1:
                st.plotly_chart(plot_3d_vowels_custom(vowel_data), use_container_width=True)
                st.info("Размер шара соответствует интенсивности (громкости) произношения.")

            with tab2:
                df = pd.DataFrame(vowel_data)
                fig_hist = px.histogram(df, x="vowel", color="vowel", title="Частота появления гласных")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with tab3:
                gender = st.radio("Пол для сравнения с нормой:", ["женщина", "мужчина"], horizontal=True)
                norms = get_russian_norms(gender)
                # Логика построения радара (упрощенно)
                fig_radar = go.Figure()
                # ... (здесь ваш код радара) ...
                st.plotly_chart(fig_radar, use_container_width=True)

            # Кнопка скачивания
            csv = pd.DataFrame(vowel_data).to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Скачать результаты (CSV)", csv, "results.csv", "text/csv")
        else:
            st.warning("Гласные не обнаружены. Попробуйте другую запись.")

if __name__ == "__main__":
    main()
