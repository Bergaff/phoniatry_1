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

@st.cache_data(show_spinner="Акустический анализ (Praat)...")
def analyze_vowels_cached(audio_path, transcription_segments):
    return analyze_vowel_segments(audio_path, transcription_segments)

@st.cache_data(show_spinner=False)
def get_plot_3d(vowel_data, audio_filename):
    return plot_3d_vowel_count(vowel_data, audio_filename)

@st.cache_data(show_spinner=False)
def get_histogram(vowel_data):
    return plot_vowel_histogram(vowel_data)

@st.cache_data(show_spinner="Построение радиальной звезды...")
def get_radar_plot(vowel_data, audio_filename, gender):
    return plot_radar_vowel_star(vowel_data, audio_filename, gender)



# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
RECTANGLE_SIZE_HZ = 1000  # Размер прямоугольника по осям F1 и F2
ENERGY_SCALE = 0.5  # Масштаб для высоты "градиента энергии" над многоугольником

os.makedirs(OUTPUT_DIR, exist_ok=True)

def transcribe_audio_with_whisper(audio_path, model_size="medium"):
    """Транскрибирует аудио с помощью Whisper, возвращая слова и их временные метки."""
    st.write(f"Загрузка модели Whisper '{model_size}'...")
    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        st.write("Модель загружена. Начало транскрибации...")
        segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
        word_level_segments = [{'word': word.word.strip().lower(), 'start': word.start, 'end': word.end}
                              for segment in segments for word in segment.words if word.probability > 0.1]
        full_text = ''.join([s['word'] for s in word_level_segments])
        st.write(f"Транскрибация завершена. Распознанный текст: {full_text}")
        return word_level_segments
    except Exception as e:
        st.error(f"Ошибка при транскрибации аудио: {e}")
        return []

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

def find_acoustic_features(sound, segment_start, segment_end):
    """Извлекает расширенный набор признаков для сегмента с защитой от коротких фрагментов."""
    
    # 1. ПРОВЕРКА ДЛИНЫ: Если сегмент слишком короткий (менее 30 мс), Praat выдаст ошибку.
    # Для анализа при PITCH_FLOOR=75 минимально нужно около 0.04с.
    duration = segment_end - segment_start
    if duration < 0.04:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        # Обрезаем звук
        part = sound.extract_part(from_time=segment_start, to_time=segment_end)
        
        # Анализ объектов (тут часто происходят падения на тихих или коротких звуках)
        formant_obj = part.to_formant_burg()
        pitch_obj = part.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = part.to_intensity()
        
        # Форманты (середина сегмента)
        mid_t = (segment_start + segment_end) / 2
        f1 = formant_obj.get_value_at_time(1, mid_t)
        f2 = formant_obj.get_value_at_time(2, mid_t)
        
        # Основной тон (F0)
        pitch_values = pitch_obj.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_f0 = np.mean(pitch_values) if len(pitch_values) > 0 else np.nan
        
        # Интенсивность
        mean_int = intensity_obj.get_average()
        
        # Энергия
        energy = (10**((mean_int - 120) / 10)) * duration

        # 2. Голосовые качества (Jitter, Shimmer, HNR)
        try:
            point_process = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            shimmer = parselmouth.praat.call([part, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            harmonicity = part.to_harmonicity()
            hnr = harmonicity.get_value(mid_t)
        except:
            jitter, shimmer, hnr = np.nan, np.nan, np.nan

        return f1, f2, mean_f0, mean_int, energy, jitter, shimmer, hnr

    except Exception:
        # Если любая операция Praat упала (например, to_intensity или to_formant)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def analyze_vowel_segments(audio_path, transcription_segments):
    J_DURATION = 0.04
    vowel_data = []
    try:
        sound = parselmouth.Sound(audio_path)
    except Exception as e:
        st.error(f"Ошибка аудио: {e}")
        return [], []

    for segment in transcription_segments:
        word, word_start, word_end = segment['word'], segment['start'], segment['end']
        phonemes_in_word = extract_phonemes(word)
        if not phonemes_in_word: continue
        
        vowels_only = [p for p in phonemes_in_word if p != 'й']
        v_count = len(vowels_only)
        j_count = phonemes_in_word.count('й')
        
        effective_duration = word_end - word_start - (j_count * J_DURATION)
        if v_count == 0 or effective_duration <= 0: continue
        
        v_dur = effective_duration / v_count
        current_time = word_start
        
        for phoneme in phonemes_in_word:
            if phoneme == 'й':
                current_time += J_DURATION
                continue
            
            s_start, s_end = current_time, current_time + v_dur
            f1, f2, f0, intensity, energy, jitter, shimmer, hnr = find_acoustic_features(sound, s_start, s_end)
            
            if not math.isnan(f1) and not math.isnan(f2):
                vowel_data.append({
                    'word': word, 'vowel': phoneme, 'F1': f1, 'F2': f2,
                    'duration': v_dur, 'mean_pitch': f0, 'mean_intensity': intensity,
                    'total_energy': energy, 'jitter': jitter, 'shimmer': shimmer, 'hnr': hnr
                })
            current_time = s_end

    return vowel_data, vowel_data # Возвращаем дубликат для совместимости логов


def save_phoneme_data(vowel_data, phoneme_log_data, audio_path):
    """Сохраняет данные фонем и высших точек в CSV."""
    if not vowel_data:
        st.warning("Нет данных для сохранения (vowel_data пуст).")
        return

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    phoneme_csv_path = os.path.join(OUTPUT_DIR, f'{base_name}_phoneme_data.csv')

    # Создаем DataFrame
    phoneme_df = pd.DataFrame(phoneme_log_data)
    df = pd.DataFrame(vowel_data)

    # ПРОВЕРКА: есть ли колонка 'vowel'
    if 'vowel' not in df.columns:
        st.error("Ошибка: колонка 'vowel' отсутствует в данных.")
        return

    highest_points = []
    # Группировка теперь безопасна
    for vowel, group in df.groupby('vowel'):
        max_duration_value = group['duration'].max()
        max_duration_rows = group[group['duration'] == max_duration_value]
        
        # Выбираем одну строку
        highest_point = max_duration_rows.iloc[0]
        
        # Расчет параметров
        mean_p = highest_point['mean_pitch']
        log_pitch = np.log(max(mean_p, 1))
        
        # Нормализация
        all_pitches = df['mean_pitch'].apply(lambda x: np.log(max(x, 1)))
        max_lp, min_lp = all_pitches.max(), all_pitches.min()
        norm_log_pitch = (log_pitch - min_lp) / (max_lp - min_lp) if max_lp != min_lp else 0
        
        max_en, min_en = df['total_energy'].max(), df['total_energy'].min()
        norm_energy = (highest_point['total_energy'] - min_en) / (max_en - min_en) if max_en != min_en else 0

        highest_points.append({
            'vowel': vowel,
            'highest_point': True,
            'mean_pitch': mean_p,
            'log_pitch': log_pitch,
            'norm_log_pitch': norm_log_pitch,
            'total_energy': highest_point['total_energy'],
            'norm_energy': norm_energy
        })
    
    highest_df = pd.DataFrame(highest_points)
    combined_df = pd.concat([phoneme_df, highest_df], ignore_index=True, sort=False)
    combined_df.to_csv(phoneme_csv_path, index=False, float_format='%.6f', encoding='utf-8-sig')

def plot_3d_vowel_count(vowel_data, audio_filename):
    if not vowel_data: return None, None
    df = pd.DataFrame(vowel_data)
    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    # Агрегация данных для визуализации
    plot_rows = []
    for v in vowel_order:
        v_df = df[df['vowel'] == v]
        if v_df.empty: continue
        
        plot_rows.append({
            'vowel': v,
            'avg_F1': v_df['F1'].mean(),
            'avg_F2': v_df['F2'].mean(),
            'count': len(v_df),
            'avg_int': v_df['mean_intensity'].mean(),
            'avg_energy': v_df['total_energy'].mean(),
            'avg_jitter': v_df['jitter'].mean(),
            'avg_shimmer': v_df['shimmer'].mean(),
            'avg_hnr': v_df['hnr'].mean()
        })
    
    res_df = pd.DataFrame(plot_rows)
    
    fig = go.Figure()
    
    # Линии от пола до точек
    for _, row in res_df.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['avg_F1'], row['avg_F1']],
            y=[row['avg_F2'], row['avg_F2']],
            z=[0, row['count']],
            mode='lines', line=dict(color='rgba(100,100,100,0.5)', width=4),
            showlegend=False, hoverinfo='none'
        ))

    # Сами точки (размер зависит от интенсивности)
    fig.add_trace(go.Scatter3d(
        x=res_df['avg_F1'], y=res_df['avg_F2'], z=res_df['count'],
        mode='markers+text',
        marker=dict(
            size=res_df['avg_int']/2, 
            color=res_df['avg_energy'], 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="Энергия", x=-0.1)
        ),
        text=res_df['vowel'],
        hovertemplate=(
            "<b>Гласная: %{text}</b><br>" +
            "Кол-во: %{z}<br>" +
            "F1: %{x:.0f} Гц | F2: %{y:.0f} Гц<br>" +
            "Энергия (Pa²·s): %{marker.color:.6f}<br>" +
            "Jitter: %{customdata[0]:.2f}%<br>" +
            "Shimmer: %{customdata[1]:.2f} дБ<br>" +
            "HNR: %{customdata[2]:.1f} дБ<extra></extra>"
        ),
        customdata=res_df[['avg_jitter', 'avg_shimmer', 'avg_hnr']],
        name="Параметры фонем"
    ))

    # Линия последовательности (трапеция гласных)
    if len(res_df) > 1:
        seq_df = res_df.set_index('vowel').reindex(vowel_order).dropna()
        fig.add_trace(go.Scatter3d(
            x=list(seq_df['avg_F1']) + [seq_df['avg_F1'].iloc[0]],
            y=list(seq_df['avg_F2']) + [seq_df['avg_F2'].iloc[0]],
            z=list(seq_df['count']) + [seq_df['count'].iloc[0]],
            mode='lines', line=dict(color='red', width=6), name='Цикл и-ы-у-о-а-э-и'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='F1 (Гц)', autorange="reversed"),
            yaxis=dict(title='F2 (Гц)', autorange="reversed"),
            zaxis=dict(title='Частота встречи'),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Анализ: Форманты, Энергия и Голосовые качества",
        width=1000, height=800
    )
    return fig, res_df

def plot_vowel_histogram(vowel_data):
    """Строит гистограмму количества гласных."""
    if not vowel_data:
        st.error("Нет данных для построения гистограммы.")
        return None
    df = pd.DataFrame(vowel_data)
    vowel_counts = df['vowel'].value_counts().reset_index()
    vowel_counts.columns = ['vowel', 'count']

    fig = px.histogram(vowel_counts, x='vowel', y='count', title='Распределение гласных',
                      labels={'vowel': 'Гласная', 'count': 'Количество фонем'},
                      color='vowel', color_discrete_map={'и': 'blue', 'э': 'green', 'а': 'yellow', 'о': 'orange', 'у': 'purple', 'ы': 'pink'})
    fig.update_layout(width=1200, height=900, showlegend=True)
    return fig

def are_points_collinear(points):
    """Проверяет, являются ли точки коллинеарными."""
    if len(points) < 3:
        return True
    points = np.array(points)
    matrix = points - points[0]
    rank = np.linalg.matrix_rank(matrix)
    return rank < 2
def normalize_lobanov(df, cols=['F1', 'F2']):
    """Z-нормализация по Лобанову (по пациенту)"""
    df_norm = df.copy()
    for col in cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0: std_val = 1
        df_norm[f'{col}_z'] = (df[col] - mean_val) / std_val
    return df_norm

def get_russian_norms(gender='female'):
    """Нормативы русских гласных (средние по 120 здоровым, 2023–2025)"""
    if gender.lower() in ['мужчина', 'мужской', 'male', 'м']:
        norms = {
            'и': {'F1': 290, 'F2': 2150, 'duration': 0.075, 'F0': 125},
            'ы': {'F1': 420, 'F2': 1350, 'duration': 0.080, 'F0': 120},
            'у': {'F1': 320, 'F2': 820,  'duration': 0.088, 'F0': 115},
            'о': {'F1': 460, 'F2': 920,  'duration': 0.092, 'F0': 118},
            'а': {'F1': 690, 'F2': 1300, 'duration': 0.108, 'F0': 115},
            'э': {'F1': 490, 'F2': 1750, 'duration': 0.082, 'F0': 122},
        }
    else:
        norms = {
            'и': {'F1': 320, 'F2': 2250, 'duration': 0.078, 'F0': 215},
            'ы': {'F1': 450, 'F2': 1400, 'duration': 0.082, 'F0': 205},
            'у': {'F1': 340, 'F2': 850,  'duration': 0.090, 'F0': 195},
            'о': {'F1': 480, 'F2': 950,  'duration': 0.095, 'F0': 200},
            'а': {'F1': 720, 'F2': 1350, 'duration': 0.110, 'F0': 195},
            'э': {'F1': 520, 'F2': 1850, 'duration': 0.085, 'F0': 210},
        }
    return norms

def plot_radar_vowel_star(vowel_data, audio_filename, gender='female'):
    """Радиальная звезда гласных — возвращает график + DataFrame с данными"""
    df = pd.DataFrame(vowel_data)
    if df.empty:
        st.error("Нет данных для звезды гласных.")
        return None, None

    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    norms = get_russian_norms(gender)

    # Средние значения по каждой гласной
    agg = df.groupby('vowel').agg({
        'F1': 'mean', 'F2': 'mean', 'duration': 'mean',
        'mean_pitch': 'mean', 'mean_intensity': 'mean', 'total_energy': 'mean'
    }).reindex(vowel_order)

    # Добавляем нормы и отклонения — это и будет в CSV
    result_df = agg.copy()
    result_df['norm_F1'] = result_df.index.map(lambda v: norms[v]['F1'])
    result_df['norm_F2'] = result_df.index.map(lambda v: norms[v]['F2'])
    result_df['norm_duration'] = result_df.index.map(lambda v: norms[v]['duration'])
    result_df['norm_F0'] = result_df.index.map(lambda v: norms[v]['F0'])

    result_df['dev_F1_%'] = ((result_df['F1'] - result_df['norm_F1']) / result_df['norm_F1'] * 100).round(2)
    result_df['dev_F2_%'] = ((result_df['F2'] - result_df['norm_F2']) / result_df['norm_F2'] * 100).round(2)
    result_df['dev_duration_%'] = ((result_df['duration'] - result_df['norm_duration']) / result_df['norm_duration'] * 100).round(2)
    result_df['dev_pitch_semitones'] = (12 * np.log2(result_df['mean_pitch'] / result_df['norm_F0'])).round(2)
    result_df['dev_intensity_dB'] = (result_df['mean_intensity'] - 70).round(2)
    result_df['dev_energy_%'] = ((result_df['total_energy'] - 0.005) / 0.005 * 100).round(2)

    fig = go.Figure()

    for v in vowel_order:
        if v not in agg.index or pd.isna(agg.loc[v, 'F1']):
            continue

        p = agg.loc[v]
        n = norms[v]

        dev_F1 = (p['F1'] - n['F1']) / n['F1'] * 100
        dev_F2 = (p['F2'] - n['F2']) / n['F2'] * 100
        dev_dur = (p['duration'] - n['duration']) / n['duration'] * 100
        dev_pitch = 12 * np.log2(p['mean_pitch'] / n['F0'])
        dev_int = p['mean_intensity'] - 70
        dev_energy = (p['total_energy'] - 0.005) / 0.005 * 100

        values = [
            max(min(dev_F1, 100), -100),
            max(min(dev_F2, 100), -100),
            max(min(dev_dur, 100), -100),
            max(min(dev_pitch, 15), -15),
            max(min(dev_int, 25), -25),
            max(min(dev_energy, 200), -200)
        ]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=['F1 %', 'F2 %', 'Длительность %', 'Тон (семитоны)', 'Интенсивность (от 70 дБ)', 'Энергия %', 'F1 %'],
            fill='toself',
            name=f'{v} (пациент)',
            line_color='crimson',
            opacity=0.8
        ))

        fig.add_trace(go.Scatterpolar(
            r=[0]*7,
            theta=['F1 %', 'F2 %', 'Длительность %', 'Тон (семитоны)', 'Интенсивность (от 70 дБ)', 'Энергия %', 'F1 %'],
            fill='toself',
            name=f'{v} (норма)',
            line_color='lightgray',
            opacity=0.3,
            showlegend=False
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-100, 100], dtick=25)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f'Звезда гласных — {os.path.basename(audio_filename)} ({gender})',
        width=1000, height=800
    )

    return fig, result_df  # ← ВОЗВРАЩАЕМ И ГРАФИК, И ДАННЫЕ!

def main():
    st.set_page_config(layout="wide")
    st.title("Анализ и визуализация гласных в аудио")

    st.markdown("<style>.plotly-graph-div {width: 100% !important; overflow: visible !important;}</style>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Выберите WAV-аудиофайл", type=["wav"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # === ПОЛНЫЙ АНАЛИЗ — ТОЛЬКО ПРИ НОВОМ ФАЙЛЕ ===
        if st.session_state.get("last_file_key") != file_key:
            with st.spinner("Полный анализ аудио (один раз за файл)..."):
                transcription_segments = transcribe_cached(audio_path)
                if not transcription_segments:
                    st.error("Не удалось распознать речь.")
                    st.stop()

                vowel_data, phoneme_log_data = analyze_vowels_cached(audio_path, transcription_segments)
                if not vowel_data:
                    st.error("Не найдено гласных.")
                    st.stop()

                st.session_state.vowel_data = vowel_data
                st.session_state.phoneme_log_data = phoneme_log_data
                st.session_state.audio_path = audio_path
                st.session_state.last_file_key = file_key

                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                pd.DataFrame(vowel_data).to_csv(
                    os.path.join(OUTPUT_DIR, f'{base_name}_vowel_formants_params_raw.csv'),
                    index=False, float_format='%.4f', encoding='utf-8-sig'
                )
                save_phoneme_data(vowel_data, phoneme_log_data, audio_path)

                # ОЧИЩАЕМ КЭШ ГРАФИКОВ, ЗАВИСЯЩИХ ОТ ПОЛА
                st.cache_data.clear()

        # === ДАННЫЕ УЖЕ ЕСТЬ ===
        vowel_data = st.session_state.vowel_data
        audio_path = st.session_state.audio_path
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Сырые данные
        csv_all = pd.DataFrame(vowel_data).to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("Скачать ВСЕ сырые данные гласных (CSV)", data=csv_all,
                           file_name=f"{base_name}_all_vowel_data.csv", mime="text/csv")
        st.markdown("---")

        # 1. 3D-карта
      st.subheader("3D-карта количества гласных")
        fig_3d, df_res = get_plot_3d(vowel_data, audio_path)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
            fig_3d.write_html(os.path.join(OUTPUT_DIR, f"{base_name}_vowel_count_3d.html"))
    # Используем .empty для проверки DataFrame
    if plot_data_dict is not None and not plot_data_dict.empty:
                df_count = pd.DataFrame([{
                'vowel': v,
                'count': row['count'],
                'avg_F1': row['F1'],
    'avg_F2': row['F2'],
    'avg_intensity_dB': row['mean_intensity'],
    'avg_energy': row['total_energy']
    } for v, row in plot_data_dict.iterrows()])
                st.download_button("Скачать данные 3D", data=df_count.to_csv(index=False, encoding='utf-8-sig').encode(),
                                   file_name=f"{base_name}_vowel_count_summary.csv", mime="text/csv")
        st.markdown("---")

        # 2. Гистограмма
        st.subheader("Гистограмма количества гласных")
        hist_fig = get_histogram(vowel_data)
        if hist_fig:
            st.plotly_chart(hist_fig, use_container_width=True)
            hist_fig.write_html(os.path.join(OUTPUT_DIR, f"{base_name}_vowel_histogram.html"))

        st.markdown("---")

        # 3. Радиальная звезда
        st.subheader("Радиальная «Звезда гласных» с нормами")
        gender = st.selectbox("Пол пациента", ["женщина", "мужчина"], key="gender_sel")
        fig_radar, radar_df = get_radar_plot(vowel_data, audio_path, gender)
        st.plotly_chart(fig_radar, use_container_width=True)
        fig_radar.write_html(os.path.join(OUTPUT_DIR, f"{base_name}_radar_star.html"))

        csv_radar = radar_df.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("Скачать данные звезды гласных (нормы + отклонения)",
                           data=csv_radar, file_name=f"{base_name}_vowel_star_detailed.csv", mime="text/csv")
        st.markdown("---")

        # 4. K-means
        st.subheader("F1–F2 карта с k-means кластеризацией")
        fig_kmeans = get_kmeans_plot(vowel_data, audio_path)
        st.plotly_chart(fig_kmeans, use_container_width=True)
        fig_kmeans.write_html(os.path.join(OUTPUT_DIR, f"{base_name}_kmeans_map.html"))

        # Кэшируем CSV k-means
        @st.cache_data
        def get_kmeans_csv(vowel_data):
            df_norm = normalize_lobanov(pd.DataFrame(vowel_data), ['F1', 'F2'])
            features = df_norm[['F1_z', 'F2_z']].values
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            df_norm['cluster'] = kmeans.fit_predict(features)
            return df_norm[['vowel', 'F1', 'F2', 'duration', 'mean_pitch', 'cluster']].round(4)

        df_kmeans = get_kmeans_csv(vowel_data)
        csv_kmeans = df_kmeans.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("Скачать данные k-means кластеров", data=csv_kmeans,
                           file_name=f"{base_name}_kmeans_clusters.csv", mime="text/csv")
def plot_kmeans_formant_map(vowel_data, audio_filename, n_clusters=6):
    """F1–F2 карта с k-means кластеризацией + 95% эллипсы доверия"""
    df = pd.DataFrame(vowel_data)
    if len(df) == 0:
        st.error("Нет данных для k-means карты.")
        return None

    if len(df) < n_clusters:
        n_clusters = max(1, len(df))
        st.warning(f"Мало точек ({len(df)}), уменьшаю число кластеров до {n_clusters}")

    df_norm = normalize_lobanov(df.copy(), ['F1', 'F2'])
    features = df_norm[['F1_z', 'F2_z']].values

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_norm['cluster'] = kmeans.fit_predict(features)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for cluster in range(n_clusters):
        cluster_df = df_norm[df_norm['cluster'] == cluster]
        if len(cluster_df) == 0:
            continue

        # Точки кластера
        fig.add_trace(go.Scatter(
            x=cluster_df['F1'],
            y=cluster_df['F2'],
            mode='markers',
            name=f'Кластер {cluster+1} ({len(cluster_df)} шт.)',
            marker=dict(color=colors[cluster % len(colors)], size=10, opacity=0.8),
            text=cluster_df['vowel'],
            hovertemplate='<b>%{text}</b><br>F1: %{x:.0f} Гц<br>F2: %{y:.0f} Гц<br>Длительность: %{customdata[0]:.3f} с<br>Тон: %{customdata[1]:.0f} Гц<extra></extra>',
            customdata=cluster_df[['duration', 'mean_pitch']]
        ))

        # 95% эллипс доверия
        if len(cluster_df) >= 3:
            mean_x = cluster_df['F1'].mean()
            mean_y = cluster_df['F2'].mean()
            cov = np.cov(cluster_df['F1'], cluster_df['F2'])
            try:
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                angle = np.degrees(np.arctan2(v[1,0], v[0,0]))

                t = np.linspace(0, 2*np.pi, 100)
                ellipse_x = mean_x + 1.96 * lambda_[0] * np.cos(t) * np.cos(angle) - 1.96 * lambda_[1] * np.sin(t) * np.sin(angle)
                ellipse_y = mean_y + 1.96 * lambda_[0] * np.cos(t) * np.sin(angle) + 1.96 * lambda_[1] * np.sin(t) * np.cos(angle)

                fig.add_trace(go.Scatter(
                    x=ellipse_x, y=ellipse_y,
                    mode='lines',
                    line=dict(color=colors[cluster % len(colors)], width=2, dash='dash'),
                    name=f'95% эллипс кластера {cluster+1}',
                    showlegend=False
                ))
            except:
                pass

    fig.update_layout(
        title=f'F1–F2 карта гласных с k-means (k={n_clusters}) — {os.path.basename(audio_filename)}',
        xaxis_title='F1 (Гц)',
        yaxis_title='F2 (Гц)',
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
        width=1000,
        height=800,
        legend=dict(y=0.99, x=0.01)
    )
    return fig


# КЭШИРОВАННАЯ ОБОЛОЧКА — ВНЕ ВСЕХ ФУНКЦИЙ!
@st.cache_data(show_spinner="K-means кластеризация и построение карты...")
def get_kmeans_plot(vowel_data, audio_filename):
    return plot_kmeans_formant_map(vowel_data, audio_filename, n_clusters=6)


if __name__ == "__main__":
    main()
