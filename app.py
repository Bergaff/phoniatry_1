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


# Добавь в самый верх, после импортов
@st.cache_resource(show_spinner="Загрузка модели Whisper (один раз)...")
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

@st.cache_data(show_spinner="Акустический анализ гласных (Praat)...")
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

@st.cache_data(show_spinner="K-means кластеризация...")
def get_kmeans_plot(vowel_data, audio_filename):
    return plot_kmeans_formant_map(vowel_data, audio_filename, n_clusters=6)


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
    """Извлекает фонемы, корректно обрабатывая йотированные гласные."""
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
    """Извлекает F1, F2, F0, интенсивность и длительность для заданного временного сегмента."""
    F1_values, F2_values, pitch_values, intensity_values = [], [], [], []
    time_step = 0.005
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
        t += time_step
    median_f1 = np.nanmedian(F1_values) if F1_values else np.nan
    median_f2 = np.nanmedian(F2_values) if F2_values else np.nan
    median_pitch = np.nanmedian(pitch_values) if pitch_values else np.nan
    median_intensity = np.nanmedian(intensity_values) if intensity_values else np.nan
    duration = segment_end - segment_start
    return median_f1, median_f2, duration, median_pitch, median_intensity

def analyze_vowel_segments(audio_path, transcription_segments):
    """Анализирует гласные и возвращает акустические характеристики."""
    J_DURATION = 0.04
    vowel_data = []
    phoneme_log_data = []
    try:
        sound = parselmouth.Sound(audio_path)
        formant_obj = sound.to_formant_burg()
        pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity()
    except Exception as e:
        st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось обработать аудиофайл: {e}")
        return [], []
    if not transcription_segments:
        st.error("Список транскрибированных сегментов пуст.")
        return [], []
    for segment in transcription_segments:
        word, word_start, word_end = segment['word'], segment['start'], segment['end']
        phonemes_in_word = extract_phonemes(word)
        if not phonemes_in_word: continue
        j_count = phonemes_in_word.count('й')
        vowel_phonemes_count = len([p for p in phonemes_in_word if p != 'й'])
        effective_duration = word_end - word_start - (j_count * J_DURATION)
        if vowel_phonemes_count == 0 or effective_duration <= 0: continue
        vowel_duration_part = effective_duration / vowel_phonemes_count
        current_time = word_start
        for phoneme in phonemes_in_word:
            if phoneme == 'й':
                current_time += J_DURATION
                continue
            vowel_segment_start = current_time
            vowel_segment_end = current_time + vowel_duration_part
            median_f1, median_f2, duration, median_pitch, median_intensity = find_acoustic_features(
                formant_obj, pitch_obj, intensity_obj, vowel_segment_start, vowel_segment_end
            )
            if not (math.isnan(median_f1) or math.isnan(median_f2) or math.isnan(duration) or math.isnan(median_pitch)):
                impulses = median_pitch * duration
                total_energy = 0.00012 * impulses - 0.00015
                vowel_data.append({
                    'word': word, 'vowel': phoneme, 'F1': median_f1, 'F2': median_f2,
                    'duration': duration, 'mean_pitch': median_pitch, 'mean_intensity': median_intensity,
                    'start_time': vowel_segment_start, 'end_time': vowel_segment_end, 'total_energy': total_energy
                })
                phoneme_log_data.append({
                    'vowel': phoneme, 'word': word, 'F1': median_f1, 'F2': median_f2,
                    'duration': duration, 'mean_pitch': median_pitch, 'mean_intensity': median_intensity,
                    'total_energy': total_energy
                })
            current_time = vowel_segment_end
    st.write(f"Всего данных о гласных собрано: {len(vowel_data)}")
    return vowel_data, phoneme_log_data

def save_phoneme_data(vowel_data, phoneme_log_data, audio_path):
    """Сохраняет данные фонем и высших точек в CSV."""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    phoneme_csv_path = os.path.join(OUTPUT_DIR, f'{base_name}_phoneme_data.csv')

    # Подготовка данных о фонемах
    phoneme_df = pd.DataFrame(phoneme_log_data)

    # Подготовка данных о высших точках
    df = pd.DataFrame(vowel_data)
    highest_points = []
    for vowel, group in df.groupby('vowel'):
        max_duration_value = group['duration'].max()
        max_duration_rows = group[group['duration'] == max_duration_value]
        highest_point = max_duration_rows.sample(n=1, random_state=random.randint(0, 1000)).iloc[0] if len(max_duration_rows) > 1 else max_duration_rows.iloc[0]
        log_pitch = np.log(max(highest_point['mean_pitch'], 1))
        max_log_pitch = df['mean_pitch'].apply(lambda x: np.log(max(x, 1))).max()
        min_log_pitch = df['mean_pitch'].apply(lambda x: np.log(max(x, 1))).min()
        log_pitch_range = max_log_pitch - min_log_pitch if max_log_pitch != min_log_pitch else 1
        norm_log_pitch = (log_pitch - min_log_pitch) / log_pitch_range
        max_energy = df['total_energy'].max()
        min_energy = df['total_energy'].min()
        energy_range = max_energy - min_energy if max_energy != min_energy else 1
        norm_energy = (highest_point['total_energy'] - min_energy) / energy_range
        highest_points.append({
            'vowel': vowel,
            'highest_point': True,
            'mean_pitch': highest_point['mean_pitch'],
            'log_pitch': log_pitch,
            'norm_log_pitch': norm_log_pitch,
            'total_energy': highest_point['total_energy'],
            'norm_energy': norm_energy
        })
    highest_df = pd.DataFrame(highest_points)

    # Объединение данных
    combined_df = pd.concat([phoneme_df, highest_df], ignore_index=True, sort=False)
    combined_df.to_csv(phoneme_csv_path, index=False, float_format='%.6f', encoding='utf-8-sig')

def plot_3d_vowel_count(vowel_data, audio_filename):
    """Создает 3D-график, соединяя пики линий в порядке и-ы-у-о-а-э-и."""
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    if not vowel_data:
        st.error("Нет данных для построения графика количества гласных.")
        return None, None

    vowel_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    df = pd.DataFrame(vowel_data)

    aggregated_data = {vowel: {'F1': [], 'F2': [], 'mean_intensity': [], 'mean_pitch': [], 'total_energy': []} for vowel in vowel_order}
    for item in vowel_data:
        vowel = item['vowel']
        if vowel in vowel_order:
            aggregated_data[vowel]['F1'].append(item['F1'])
            aggregated_data[vowel]['F2'].append(item['F2'])
            aggregated_data[vowel]['mean_intensity'].append(item['mean_intensity'])
            aggregated_data[vowel]['mean_pitch'].append(item['mean_pitch'])
            aggregated_data[vowel]['total_energy'].append(item['total_energy'])

    plot_data_dict = {}
    for vowel in vowel_order:
        if aggregated_data[vowel]['F1']:
            plot_data_dict[vowel] = {
                'avg_F1': np.mean(aggregated_data[vowel]['F1']),
                'avg_F2': np.mean(aggregated_data[vowel]['F2']),
                'avg_intensity': np.mean(aggregated_data[vowel]['mean_intensity']),
                'avg_pulses': np.mean(aggregated_data[vowel]['mean_pitch']) * np.mean([item['duration'] for item in vowel_data if item['vowel'] == vowel]),
                'avg_energy': np.mean(aggregated_data[vowel]['total_energy']),
                'count': len(aggregated_data[vowel]['F1'])
            }

    if not plot_data_dict:
        st.error("Нет данных для построения графика количества гласных.")
        return None, None

    x_coords, y_coords, z_heights, vowel_labels, marker_sizes = [], [], [], [], []
    hover_texts = []

    all_intensities = [data['avg_intensity'] for data in plot_data_dict.values()]
    min_intensity = min(all_intensities) if all_intensities else 0
    max_intensity = max(all_intensities) if all_intensities else 1

    def normalize_intensity(val, min_val, max_val, scale_min=10, scale_max=40):
        if max_val == min_val: return scale_min
        return scale_min + (val - min_val) / (max_val - min_val) * (scale_max - scale_min)

    for vowel in vowel_order:
        if vowel in plot_data_dict and plot_data_dict[vowel]:
            data = plot_data_dict[vowel]
            x_coords.append(data['avg_F1'])
            y_coords.append(data['avg_F2'])
            z_heights.append(data['count'])
            vowel_labels.append(vowel)
            marker_sizes.append(normalize_intensity(data['avg_intensity'], min_intensity, max_intensity))

            hover_text = f"Фонема: {vowel}<br>Количество: {data['count']}<br>F1: {data['avg_F1']:.0f} Гц<br>F2: {data['avg_F2']:.0f} Гц<br>Интенсивность: {data['avg_intensity']:.1f} дБ<br>Энергия: {data['avg_energy']:.6f} Pa²·sec<br>Импульсы: {data['avg_pulses']:.0f}"
            hover_texts.append(hover_text)

    if len(x_coords) > 0:
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        z_heights.append(z_heights[0])

    lines_list = []
    for i in range(len(vowel_labels)):
        x, y, z = x_coords[i], y_coords[i], z_heights[i]
        line = go.Scatter3d(
            x=[x, x], y=[y, y], z=[0, z], mode='lines',
            line=dict(color='gray', width=5),
            hoverinfo='none',
            showlegend=False
        )
        lines_list.append(line)

    scatter_plot_base = go.Scatter3d(
        x=x_coords[:-1],
        y=y_coords[:-1],
        z=[0] * len(x_coords[:-1]),
        mode='markers+text',
        marker=dict(
            size=marker_sizes,
            color=np.arange(len(x_coords[:-1])),
            colorscale='Viridis',
            sizemode='diameter',
            sizeref=max(marker_sizes) / 50 if max(marker_sizes) > 0 else 1
        ),
        text=[f"Фонема: {v}" for v in vowel_labels],
        textposition="bottom center",
        hoverinfo='text',
        hovertext=hover_texts,
        name='Гласные фонемы'
    )

    connecting_line = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_heights,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=5, color='red'),
        name='Последовательность и-ы-у-о-а-э-и',
        hoverinfo='none'
    )

    fig = go.Figure(data=lines_list + [scatter_plot_base, connecting_line])

    max_z = max(z_heights) if z_heights else 1
    fig.update_layout(
        title=f'3D-карта количества гласных фонем - {base_name}',
        scene=dict(
            xaxis_title='Форманта F1 (Гц)',
            yaxis_title='Форманта F2 (Гц)',
            zaxis_title='Количество фонем',
            xaxis=dict(autorange="reversed"),
            yaxis=dict(autorange="reversed"),
            zaxis=dict(range=[0, max_z + 1])
        ),
        width=1200, height=900, showlegend=True
    )
    return fig, plot_data_dict

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
        fig_3d, plot_data_dict = get_plot_3d(vowel_data, audio_path)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
            fig_3d.write_html(os.path.join(OUTPUT_DIR, f"{base_name}_vowel_count_3d.html"))
            if plot_data_dict:
                df_count = pd.DataFrame([{
                    'vowel': v, 'count': d['count'],
                    'avg_F1': d['avg_F1'], 'avg_F2': d['avg_F2'],
                    'avg_intensity_dB': d['avg_intensity'], 'avg_energy': d['avg_energy']
                } for v, d in plot_data_dict.items()])
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

if __name__ == "__main__":
    main()
