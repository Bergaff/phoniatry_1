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
import streamlit as st
from sklearn.cluster import KMeans

# --- Константы ---
OUTPUT_DIR = "./SpeechViz3D"
WHISPER_MODEL = "medium"
PITCH_FLOOR = 75
PITCH_CEILING = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Кэширование ---
@st.cache_resource(show_spinner="Загрузка модели Whisper...")
def load_whisper_model():
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

@st.cache_data(show_spinner="Транскрибация...")
def transcribe_cached(audio_path):
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="ru")
    word_level_segments = []
    full_text = []
    for segment in segments:
        for word in segment.words:
            if word.probability > 0.1:
                word_level_segments.append({'word': word.word.strip().lower(), 'start': word.start, 'end': word.end})
                full_text.append(word.word.strip())
    return word_level_segments, " ".join(full_text)

# --- Функции анализа (без изменений) ---
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
    f1_l, f2_l, p_l, i_l = [], [], [], []
    t = start
    while t < end:
        f1 = formant_obj.get_value_at_time(1, t)
        f2 = formant_obj.get_value_at_time(2, t)
        pitch = pitch_obj.get_value_at_time(t)
        inten = intensity_obj.get_value(t)
        if not math.isnan(f1): f1_l.append(f1)
        if not math.isnan(f2): f2_l.append(f2)
        if not math.isnan(pitch): p_l.append(pitch)
        if not math.isnan(inten): i_l.append(inten)
        t += 0.005
    return np.nanmedian(f1_l), np.nanmedian(f2_l), end-start, np.nanmedian(p_l), np.nanmedian(i_l)

def analyze_vowel_segments(audio_path, transcription_segments):
    sound = parselmouth.Sound(audio_path)
    formant_obj = sound.to_formant_burg()
    pitch_obj = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    intensity_obj = sound.to_intensity()
    
    vowel_data = []
    for seg in transcription_segments:
        word = seg['word']
        phonemes = extract_phonemes(word)
        v_only = [p for p in phonemes if p != 'й']
        if not v_only: continue
        
        dur_v = (seg['end'] - seg['start'] - (phonemes.count('й') * 0.04)) / len(v_only)
        curr = seg['start']
        for p in phonemes:
            if p == 'й':
                curr += 0.04
                continue
            f1, f2, d, pitch, intens = find_acoustic_features(
                formant_obj, pitch_obj, intensity_obj, curr, curr + dur_v)
            if not np.isnan([f1, f2, pitch, intens]).any():
                energy = intens * d * 0.001
                vowel_data.append({
                    'word': word, 'vowel': p, 'F1': f1, 'F2': f2,
                    'duration': d, 'mean_pitch': pitch, 'mean_intensity': intens,
                    'total_energy': energy
                })
            curr += dur_v
    return vowel_data

# --- 3D-карта ---
def plot_3d_vowel_map(vowel_data):
    df = pd.DataFrame(vowel_data)
    v_order = ['и', 'ы', 'у', 'о', 'а', 'э']
    
    stats = []
    for v in v_order:
        v_df = df[df['vowel'] == v]
        if v_df.empty: continue
        stats.append({
            'vowel': v,
            'count': len(v_df),
            'F1': v_df['F1'].mean(),
            'F2': v_df['F2'].mean(),
            'mean_energy': v_df['total_energy'].mean(),
            'mean_pitch': v_df['mean_pitch'].mean(),
            'mean_intensity': v_df['mean_intensity'].mean()
        })
    
    pdf = pd.DataFrame(stats)
    if pdf.empty:
        return go.Figure(), pd.DataFrame(), df

    # УМЕНЬШЕННЫЙ масштаб шариков
    e_min, e_max = pdf['mean_energy'].min(), pdf['mean_energy'].max()
    if e_max == e_min:
        sizes = [14] * len(pdf)
    else:
        sizes = 10 + 22 * (pdf['mean_energy'] - e_min) / (e_max - e_min)   # 10–32 px

    fig = go.Figure()

    colors = np.arange(len(pdf))

    for i, row in pdf.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['F1'], row['F1']],
            y=[row['F2'], row['F2']],
            z=[0, row['count']],
            mode='lines',
            line=dict(color='rgba(180,180,180,0.4)', width=2.5),
            hoverinfo='skip',
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[row['F1']],
            y=[row['F2']],
            z=[0],
            mode='markers+text',
            text=[f'"{row["vowel"]}"'],
            textposition="middle center",
            marker=dict(
                size=sizes[i],
                color=colors[i],
                colorscale='Plasma',
                opacity=0.92,
                line=dict(color='rgba(30,30,30,0.7)', width=1)
            ),
            name=row['vowel'],
            hovertemplate=(
                f'<b>"{row["vowel"]}"</b><br>'
                f'F1: %{{x:.0f}} Гц<br>F2: %{{y:.0f}} Гц<br>'
                f'Количество: {row["count"]}<br>'
                f'Сред. энергия: {row["mean_energy"]:.4f}<extra></extra>'
            )
        ))

    if len(pdf) >= 2:
        ldf = pd.concat([pdf, pdf.iloc[[0]]])
        fig.add_trace(go.Scatter3d(
            x=ldf['F1'], y=ldf['F2'], z=ldf['count'],
            mode='lines',
            line=dict(color='red', width=4.5),
            name='Форма пространства',
            hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='F1 (Гц)', yaxis_title='F2 (Гц)', zaxis_title='Кол-во',
            xaxis=dict(autorange="reversed"),
            yaxis=dict(autorange="reversed"),
            zaxis=dict(range=[0, max(10, pdf['count'].max() * 1.25)])
        ),
        height=680,
        margin=dict(l=5, r=5, b=5, t=30)
    )

    return fig, pdf, df

# --- Кластеризация ---
def plot_clustering_hulls(vowel_data):
    df = pd.DataFrame(vowel_data)
    if len(df) < 8:
        fig = go.Figure()
        fig.update_layout(title="Недостаточно данных для кластеризации")
        return fig
    
    df_n = df.copy()
    for c in ['F1', 'F2']:
        df_n[c] = (df[c] - df[c].mean()) / df[c].std()
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df_n[['F1', 'F2']])
    
    fig = go.Figure()
    colors = px.colors.qualitative.Safe
    
    for i in range(6):
        c_df = df[df['cluster'] == i]
        if len(c_df) < 3: continue
        
        points = c_df[['F1', 'F2']].values
        hull = ConvexHull(points)
        h_pts = points[hull.vertices]
        h_pts = np.vstack([h_pts, h_pts[0]])
        
        fig.add_trace(go.Scatter(
            x=h_pts[:,0], y=h_pts[:,1],
            fill="toself", fillcolor=colors[i], opacity=0.15,
            line=dict(color=colors[i]), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=c_df['F1'], y=c_df['F2'],
            mode='markers', marker=dict(color=colors[i], size=8),
            name=f'Кластер {i+1}', text=c_df['vowel']
        ))

    fig.update_layout(
        xaxis=dict(autorange="reversed", title="F1"),
        yaxis=dict(autorange="reversed", title="F2"),
        title="Кластеризация (Convex Hull)",
        height=600
    )
    return fig

# --- Главное приложение ---
def main():
    st.set_page_config(layout="wide", page_title="SpeechViz3D — Карта гласных")
    st.title("Анализ гласных — 3D карта")

    uploaded_file = st.file_uploader("Выберите .wav файл", type=["wav"])

    if not uploaded_file:
        return

    audio_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Обработка аудио..."):
        segments, full_text = transcribe_cached(audio_path)
        vowel_data = analyze_vowel_segments(audio_path, segments)

    if not vowel_data:
        st.error("Не удалось найти гласные в записи")
        return

    st.subheader("Транскрипция")
    st.write(full_text)
    st.markdown(f"**Найдено реализаций гласных: {len(vowel_data)}**")

    # ── Основные вкладки ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "3D Карта гласных",
        "Гистограмма",
        "Радиальный график",
        "Кластеризация"
    ])

    # Подготовка данных один раз
    fig3d, summary_df, full_df = plot_3d_vowel_map(vowel_data)

    with tab1:
        st.plotly_chart(fig3d, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Все реализации (подробно)",
                full_df.to_csv(index=False).encode('utf-8-sig'),
                "all_vowel_measurements.csv", "text/csv"
            )
        with col2:
            st.download_button(
                "📊 Средние по фонемам",
                summary_df.to_csv(index=False).encode('utf-8-sig'),
                "vowels_summary.csv", "text/csv"
            )

    with tab2:
        counts = pd.DataFrame(vowel_data)['vowel'].value_counts().reset_index(name='count')
        fig_hist = px.bar(counts, x='vowel', y='count', color='vowel',
                         title="Распределение гласных", text_auto=True)
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.download_button(
            "📥 Данные гистограммы (CSV)",
            counts.to_csv(index=False).encode('utf-8-sig'),
            "histogram_counts.csv", "text/csv"
        )

    with tab3:
        st.info("Радиальный график (Radar/Spider chart) — пока заглушка.\n"
                "Можно реализовать по средним значениям F1, F2, pitch, intensity и т.д.")
        # Здесь можно вставить код радара позже
        st.write("Пример: сравнение средних характеристик гласных в полярных координатах")

    with tab4:
        fig_cluster = plot_clustering_hulls(vowel_data)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        if len(full_df) >= 8:
            st.download_button(
                "📥 Данные для кластеризации",
                full_df.to_csv(index=False).encode('utf-8-sig'),
                "clustering_data.csv", "text/csv"
            )

if __name__ == "__main__":
    main()
