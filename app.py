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
    # word_timestamps=True позволяет получить точное время каждого слова
    segments, info = model.transcribe(audio_path, word_timestamps=True, language="ru")
    
    word_level_segments = []
    full_text_parts = []
    
    for segment in segments:
        full_text_parts.append(segment.text)
        for word in segment.words:
            # Снижаем порог вероятности до 0.05, чтобы не терять слова
            if word.probability > 0.05:
                word_level_segments.append({
                    'word': word.word.strip().lower(),
                    'start': word.start,
                    'end': word.end
                })
    
    return word_level_segments, " ".join(full_text_parts)

# --- Функции анализа ---

def extract_phonemes(text):
    """Преобразует текст в список фонем (гласные + й)."""
    phonemes = []
    # Очистка текста от мусора
    text_clean = re.sub(r'[^а-яё]', '', text.lower())
    for i, char in enumerate(text_clean):
        # Обработка йотированных гласных
        if char in 'еёюя':
            # Если в начале слова или после другой гласной/ь/ъ -> добавляем 'й'
            if i == 0 or text_clean[i-1] in 'аоуэыиеёюяьъ':
                phonemes.append('й')
            
            # Заменяем саму гласную на чистый звук
            mapping = {'е': 'э', 'ё': 'о', 'ю': 'у', 'я': 'а'}
            phonemes.append(mapping[char])
        elif char in 'аоуэыи':
            phonemes.append(char)
        elif char == 'й':
            phonemes.append('й')
    return phonemes

def find_acoustic_features(sound, segment_start, segment_end):
    """Безопасное извлечение признаков через Praat."""
    duration = segment_end - segment_start
    # Защита от слишком коротких сегментов
    if duration < 0.04:
        return [np.nan] * 8

    try:
        part = sound.extract_part(from_time=segment_start, to_time=segment_end)
        
        # Анализ
        formant_obj = part.to_formant_burg()
        pitch_obj = part.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = part.to_intensity()
        
        mid_t = (segment_start + segment_end) / 2
        f1 = formant_obj.get_value_at_time(1, mid_t)
        f2 = formant_obj.get_value_at_time(2, mid_t)
        
        # Pitch
        pitch_values = pitch_obj.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_f0 = np.mean(pitch_values) if len(pitch_values) > 0 else np.nan
        
        # Интенсивность и Энергия
        mean_int = intensity_obj.get_average()
        energy = (10**((mean_int - 120) / 10)) * duration

        # Качество голоса
        try:
            point_process = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            shimmer = parselmouth.praat.call([part, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            hnr = part.to_harmonicity().get_value(mid_t)
        except:
            jitter, shimmer, hnr = np.nan, np.nan, np.nan

        return f1, f2, mean_f0, mean_int, energy, jitter, shimmer, hnr
    except:
        return [np.nan] * 8

def analyze_vowel_segments(audio_path, transcription_segments):
    """Распределяет время слова между его фонемами и анализирует их."""
    J_DURATION = 0.04
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
    except Exception as e:
        st.error(f"Ошибка загрузки аудио в Praat: {e}")
        return []

    for segment in transcription_segments:
        word, w_start, w_end = segment['word'], segment['start'], segment['end']
        phonemes_in_word = extract_phonemes(word)
        
        # Считаем гласные (без 'й')
        vowels_only = [p for p in phonemes_in_word if p != 'й']
        if not vowels_only: continue
        
        # Вычитаем время на 'й', остальное делим поровну между гласными
        j_count = phonemes_in_word.count('й')
        useful_time = (w_end - w_start) - (j_count * J_DURATION)
        v_dur = max(useful_time / len(vowels_only), 0.04)
        
        curr_time = w_start
        for p in phonemes_in_word:
            if p == 'й':
                curr_time += J_DURATION
                continue
            
            s_start, s_end = curr_time, curr_time + v_dur
            res = find_acoustic_features(sound, s_start, s_end)
            
            if not np.isnan(res[0]): # Если F1 найден
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': res[0], 'F2': res[1],
                    'duration': v_dur, 'mean_pitch': res[2], 'mean_intensity': res[3],
                    'total_energy': res[4], 'jitter': res[5], 'shimmer': res[6], 'hnr': res[7]
                })
            curr_time = s_end
            
    return vowel_data

# --- Визуализация (Plotly) ---

def plot_3d_vowel_count(vowel_data):
    if not vowel_data: return None, None
    df = pd.DataFrame(vowel_data)
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    plot_rows = []
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if v_df.empty: continue
        plot_rows.append({
            'vowel': v, 'avg_F1': v_df['F1'].mean(), 'avg_F2': v_df['F2'].mean(),
            'count': len(v_df), 'avg_int': v_df['mean_intensity'].mean(),
            'avg_energy': v_df['total_energy'].mean(), 'avg_jitter': v_df['jitter'].mean(),
            'avg_shimmer': v_df['shimmer'].mean(), 'avg_hnr': v_df['hnr'].mean()
        })
    
    res_df = pd.DataFrame(plot_rows)
    fig = go.Figure()

    # Линии
    for _, row in res_df.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['avg_F1'], row['avg_F1']], y=[row['avg_F2'], row['avg_F2']], z=[0, row['count']],
            mode='lines', line=dict(color='rgba(100,100,100,0.5)', width=4), showlegend=False
        ))

    # Точки
    fig.add_trace(go.Scatter3d(
        x=res_df['avg_F1'], y=res_df['avg_F2'], z=res_df['count'],
        mode='markers+text',
        marker=dict(size=res_df['avg_int']/3, color=res_df['avg_energy'], colorscale='Viridis', showscale=True),
        text=res_df['vowel'],
        customdata=res_df[['avg_jitter', 'avg_shimmer', 'avg_hnr']],
        hovertemplate="<b>%{text}</b><br>Кол-во: %{z}<br>F1: %{x:.0f}<br>F2: %{y:.0f}<extra></extra>"
    ))

    fig.update_layout(scene=dict(xaxis=dict(title='F1', autorange="reversed"), 
                                 yaxis=dict(title='F2', autorange="reversed")),
                      title="3D Анализ фонем")
    return fig, res_df

def get_russian_norms(gender='female'):
    if gender == 'мужчина':
        return {'и':{'F1':290,'F2':2150,'dur':0.075,'F0':125}, 'ы':{'F1':420,'F2':1350,'dur':0.08,'F0':120},
                'у':{'F1':320,'F2':820,'dur':0.088,'F0':115}, 'о':{'F1':460,'F2':920,'dur':0.092,'F0':118},
                'а':{'F1':690,'F2':1300,'dur':0.108,'F0':115}, 'э':{'F1':490,'F2':1750,'dur':0.082,'F0':122}}
    return {'и':{'F1':320,'F2':2250,'dur':0.078,'F0':215}, 'ы':{'F1':450,'F2':1400,'dur':0.082,'F0':205},
            'у':{'F1':340,'F2':850,'dur':0.09,'F0':195}, 'о':{'F1':480,'F2':950,'dur':0.095,'F0':200},
            'а':{'F1':720,'F2':1350,'dur':0.11,'F0':195}, 'э':{'F1':520,'F2':1850,'dur':0.085,'F0':210}}

def plot_radar_star(vowel_data, gender):
    df = pd.DataFrame(vowel_data)
    norms = get_russian_norms(gender)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    agg = df.groupby('vowel').mean().reindex(v_order)
    fig = go.Figure()

    for v in v_order:
        if v not in agg.index or pd.isna(agg.loc[v, 'F1']): continue
        
        p, n = agg.loc[v], norms[v]
        # Расчет отклонений в %
        vals = [
            (p['F1']-n['F1'])/n['F1']*100, (p['F2']-n['F2'])/n['F2']*100,
            (p['duration']-n['dur'])/n['dur']*100, 12*np.log2(p['mean_pitch']/n['F0'])
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=['F1 %', 'F2 %', 'Длительность %', 'Тон (полутона)', 'F1 %'],
            fill='toself', name=f'Гласная {v}'
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-100, 100])), title="Звезда отклонений от нормы")
    return fig, agg

# --- Основное приложение Streamlit ---

def main():
    st.set_page_config(page_title="Phoniatry Analytics", layout="wide")
    st.title("🎙 Анализ акустических параметров речи")

    uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])

    if uploaded_file:
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 1. Транскрибация
        with st.status("Обработка аудио...") as status:
            st.write("Шаг 1: Распознавание речи...")
            segments, full_text = transcribe_cached(audio_path)
            st.info(f"Текст: {full_text}")
            
            st.write("Шаг 2: Акустический анализ...")
            vowel_data = analyze_vowel_segments(audio_path, segments)
            status.update(label="Анализ завершен!", state="complete")

        if not vowel_data:
            st.error("Гласные не найдены. Попробуйте другую запись.")
            return

        # --- Визуализация ---
        tab1, tab2, tab3 = st.tabs(["📊 3D Карта", "🌟 Звезда Норм", "📈 Статистика"])

        with tab1:
            fig3d, res_df = plot_3d_vowel_count(vowel_data)
            st.plotly_chart(fig3d, use_container_width=True)
            st.download_button("Скачать CSV (Сводка)", res_df.to_csv().encode('utf-8'), "summary.csv")

        with tab2:
            gender = st.radio("Пол для сравнения с нормой:", ["женщина", "мужчина"])
            fig_radar, radar_data = plot_radar_star(vowel_data, gender)
            st.plotly_chart(fig_radar, use_container_width=True)

        with tab3:
            df = pd.DataFrame(vowel_data)
            st.dataframe(df.describe().T)
            st.download_button("Скачать ВСЕ данные (сырые)", df.to_csv().encode('utf-8'), "raw_data.csv")

if __name__ == "__main__":
    main()
