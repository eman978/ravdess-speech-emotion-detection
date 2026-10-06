from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE / "ravdess_emotion_model.joblib")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(BASE / "ravdess_test_metrics.json")))
CLASSES = ["neutral", "happy", "sad", "angry"]
COLORS = {"angry": "#ef4444", "happy": "#f59e0b", "neutral": "#71717a", "sad": "#0284c7"}
CUES = {
    "angry": "Tense, emphatic delivery",
    "happy": "Brighter, more energetic delivery",
    "neutral": "Steadier, more even delivery",
    "sad": "Quieter, lower-energy delivery",
}

st.set_page_config(page_title="VoxSense | Speech Emotion AI", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --ink:#18181b; --muted:#71717a; --blue:#2563eb; --line:#e4e4e7; --paper:#ffffff; }
html, body, [class*="css"], .stApp { font-family:'DM Sans',sans-serif; }
.stApp { background:#f7f8fc; color:var(--ink); }
.block-container { max-width:1420px; padding:2.2rem 2.2rem 3rem; }
h1,h2,h3,h4 { font-family:'Manrope','DM Sans',sans-serif; letter-spacing:-.025em; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#09090b,#171923 72%,#101e3f); border-right:1px solid #27272a; }
[data-testid="stSidebar"] * { color:#e4e4e7; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap:.25rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label { padding:.55rem .8rem; border-radius:10px; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#27272a; }
.brand { display:flex; align-items:center; gap:.8rem; padding:.4rem .2rem 1.1rem; }
.logo { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center; gap:3px; background:linear-gradient(135deg,#60a5fa,#1d4ed8); }
.logo i { display:block; width:4px; border-radius:3px; background:white; }
.brand b { display:block; font:800 1.15rem Manrope,sans-serif; color:white; }
.brand span { color:#a1a1aa; font-size:.69rem; letter-spacing:.11em; text-transform:uppercase; }
.side-card,.card,.kpi { background:white; border:1px solid var(--line); border-radius:16px; box-shadow:0 2px 7px rgba(20,30,60,.035); }
.side-card { margin-top:1.25rem; padding:1rem; background:#18181b; border-color:#303038; }
.side-card .small { color:#a1a1aa; text-transform:uppercase; letter-spacing:.1em; font-size:.67rem; }
.side-card .big { font:800 1.8rem Manrope,sans-serif; color:#fff; }
.side-card .side-row { display:flex; justify-content:space-between; border-top:1px solid #303038; margin-top:.55rem; padding-top:.55rem; font-size:.78rem; color:#a1a1aa; }
.side-card .side-row b { color:#fff; }
.hero { padding:2.7rem 2.5rem; color:white; border-radius:22px; margin:.1rem 0 1.3rem; background:radial-gradient(circle at 86% 20%,rgba(59,130,246,.5),transparent 42%),linear-gradient(125deg,#09090b,#171923 58%,#1d3b80); }
.hero .tag { display:inline-block; color:#bfdbfe; background:#1d4ed855; border:1px solid #60a5fa66; border-radius:999px; padding:.28rem .75rem; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
.hero h1 { color:white; margin:.8rem 0 .5rem; font-size:2.6rem; line-height:1.12; }
.hero p { max-width:680px; color:#d4d4d8; font-size:1rem; }
.pill { display:inline-block; margin:.4rem .35rem 0 0; padding:.27rem .75rem; border-radius:999px; color:white; font-weight:700; font-size:.78rem; }
.page-head h2 { margin:0 0 .15rem; font-size:1.7rem; }
.page-head p { color:var(--muted); margin:0 0 1.1rem; }
.kpi { padding:1rem 1.1rem; border-top:3px solid var(--blue); }
.kpi .value { font:800 1.8rem Manrope,sans-serif; line-height:1.1; }
.kpi .label { color:var(--muted); font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; margin-top:.3rem; }
.card { padding:1.1rem 1.2rem; margin:.3rem 0 .8rem; }
.card h4 { margin:0 0 .4rem; font-size:1rem; }
.card p { margin:0; color:#52525b; line-height:1.5; }
.section { font:800 1.12rem Manrope,sans-serif; margin:1.2rem 0 .65rem; }
.result { border-radius:18px; padding:1.35rem 1.4rem; color:#fff; margin:.25rem 0 .8rem; }
.result .eyebrow { font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; opacity:.86; }
.result .emotion { color:white; font:800 2.4rem Manrope,sans-serif; margin:.15rem 0 .35rem; }
.result .confidence { opacity:.95; }
.prob-row { display:flex; align-items:center; gap:.7rem; margin:.55rem 0; }
.prob-name { width:85px; font-weight:700; }
.prob-track { flex:1; background:#e4e4e7; height:11px; border-radius:99px; overflow:hidden; }
.prob-fill { height:100%; border-radius:99px; }
.prob-value { width:45px; text-align:right; font-variant-numeric:tabular-nums; color:#52525b; }
.note { border-left:4px solid #2563eb; background:#eff6ff; color:#1e3a8a; padding:.75rem 1rem; border-radius:8px; margin:.5rem 0 1rem; }
.empty { text-align:center; padding:2rem 1rem; border:1px dashed #cbd5e1; border-radius:16px; background:#fff; color:#64748b; }
@media(max-width:760px) { .block-container{padding:1.5rem 1rem 2rem}.hero{padding:1.7rem 1.2rem}.hero h1{font-size:1.9rem} }
</style>
""", unsafe_allow_html=True)

for key, value in [("history", []), ("seen", set()), ("last_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def load_bundle(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        return {"_load_error": str(exc)}


@st.cache_data
def load_metrics(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def summarize_over_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.concatenate([values.mean(axis=1), values.std(axis=1)])


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Self-contained extraction matching the Kaggle notebook; no features.py needed."""
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))
    n_fft, hop = 2048, 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64),
        ref=np.max,
    )
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spectral = np.vstack([
        librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0],
    ])
    return np.concatenate([
        summarize_over_time(mfcc), summarize_over_time(delta_mfcc),
        summarize_over_time(chroma), summarize_over_time(mel_db),
        summarize_over_time(contrast), summarize_over_time(spectral),
    ]).astype(np.float32)


def display_probabilities(probabilities: dict[str, float], classes: list[str], highlight: str):
    rows = []
    for emotion in classes:
        value = float(probabilities.get(emotion, 0.0))
        opacity = "1" if emotion == highlight else ".55"
        rows.append(
            f'<div class="prob-row"><div class="prob-name">{emotion.title()}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{value * 100:.1f}%;'
            f'background:{COLORS.get(emotion, "#2563eb")};opacity:{opacity}"></div></div>'
            f'<div class="prob-value">{value:.0%}</div></div>'
        )
    st.markdown('<div class="card"><h4>Probability per emotion</h4>' + "".join(rows) + '</div>', unsafe_allow_html=True)


def analyse_audio(audio_bytes: bytes, source: str, bundle: dict, classes: list[str]):
    target_sr = int(bundle.get("sample_rate", 22_050))
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable.")
    features = extract_features(y, sr)
    expected = len(bundle.get("feature_columns", []))
    if expected and len(features) != expected:
        raise ValueError(f"This app extracted {len(features)} features, but the saved model expects {expected}. Use the matching Kaggle notebook export.")

    model = bundle["model"]
    X = features.reshape(1, -1)
    predicted = str(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        model_classes = (model.classes_ if hasattr(model, "classes_")
                         else model.named_steps["classifier"].classes_)
        probability_map = {str(name): float(value) for name, value in zip(model_classes, probs)}
    else:
        probability_map = {name: float(name == predicted) for name in classes}
    duration = len(y) / sr
    fig, ax = plt.subplots(figsize=(10, 2.6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=COLORS.get(predicted, "#2563eb"))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Uploaded speech waveform")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return {
        "digest": hashlib.sha256(audio_bytes).hexdigest(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "emotion": predicted,
        "probabilities": probability_map,
        "confidence": float(probability_map.get(predicted, 0.0)),
        "duration": duration,
        "waveform": fig,
    }


def save_history(result: dict):
    digest = result["digest"]
    if digest in st.session_state.seen:
        return
    st.session_state.seen.add(digest)
    row = {
        "Time": result["time"], "Source": result["source"],
        "Emotion": result["emotion"].title(), "Confidence": f"{result['confidence']:.0%}",
    }
    for emotion, value in result["probabilities"].items():
        row[emotion.title()] = f"{value:.0%}"
    st.session_state.history.insert(0, row)


def confusion_figure(confusion: dict):
    labels = confusion.get("labels", CLASSES)
    matrix = np.asarray(confusion.get("counts", []), dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        return None
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, values, title, fmt in [(axes[0], matrix, "Counts", ".0f"), (axes[1], normalized, "Row-normalized", ".2f")]:
        image = ax.imshow(values, cmap="Blues", vmin=0)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), [x.title() for x in labels], rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), [x.title() for x in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        color="white" if values[i, j] > values.max() * .55 else "#18181b", fontsize=9)
    fig.tight_layout()
    return fig


def kpi(value, label):
    st.markdown(f'<div class="kpi"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)


bundle = load_bundle(str(MODEL_PATH))
metrics = load_metrics(str(METRICS_PATH))
model_error = bundle.get("_load_error") if isinstance(bundle, dict) else None
if model_error:
    bundle = None
classes = bundle.get("emotion_order", CLASSES) if isinstance(bundle, dict) else CLASSES

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo"><i style="height:10px"></i><i style="height:20px"></i><i style="height:14px"></i><i style="height:23px"></i><i style="height:12px"></i></div><div><b>VoxSense</b><span>Speech emotion AI</span></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Home", "Analyze", "History", "Dashboard", "About"], label_visibility="collapsed")
    st.markdown('<div class="side-card"><div class="small">Held-out actor accuracy</div>', unsafe_allow_html=True)
    acc = metrics.get("test_accuracy") if metrics and not metrics.get("_load_error") else None
    st.markdown(f'<div class="big">{acc:.1%}</div>' if acc is not None else '<div class="big">—</div>', unsafe_allow_html=True)
    macro = metrics.get("test_macro_f1") if metrics and not metrics.get("_load_error") else None
    st.markdown(f'<div class="side-row"><span>Macro F1</span><b>{macro:.3f}</b></div>' if macro is not None else '<div class="side-row"><span>Macro F1</span><b>—</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-row"><span>Predictions this session</span><b>{len(st.session_state.history)}</b></div></div>', unsafe_allow_html=True)

if model_error:
    st.error(f"Could not load the model bundle: {model_error}")
elif bundle is None:
    st.warning(f"Model not found at {MODEL_PATH}. Add ravdess_emotion_model.joblib beside app.py.")
else:
    saved_version = bundle.get("sklearn_version")
    if saved_version and saved_version != sklearn.__version__:
        st.warning(f"Model was saved with scikit-learn {saved_version}; this app uses {sklearn.__version__}. If loading or prediction fails, match the training version.")
if metrics and metrics.get("_load_error"):
    st.warning(f"Could not read metrics JSON: {metrics['_load_error']}")

if page == "Home":
    pills = "".join(f'<span class="pill" style="background:{COLORS.get(c, "#2563eb")}">{c.title()}</span>' for c in classes)
    st.markdown('<div class="hero"><span class="tag">AI-powered speech analysis</span><h1>Hear the emotion behind every voice</h1><p>Upload a short WAV clip and explore the emotion predicted from its acoustic features.</p>' + pills + '</div>', unsafe_allow_html=True)
    if bundle is None:
        st.info("Add the Kaggle-exported model bundle to enable live predictions.")
    cols = st.columns(4)
    values = [
        (f"{metrics['test_accuracy']:.1%}", "Test accuracy") if metrics and metrics.get("test_accuracy") is not None else ("—", "Test accuracy"),
        (f"{metrics['test_macro_f1']:.3f}", "Macro F1") if metrics and metrics.get("test_macro_f1") is not None else ("—", "Macro F1"),
        (str(len(classes)), "Emotions"),
        (str(metrics.get("test_clips", "—")) if metrics else "—", "Held-out clips"),
    ]
    for col, (value, label) in zip(cols, values):
        with col:
            kpi(value, label)
    st.markdown('<div class="section">How it works</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, number, title, body in [
        (a, "01", "Upload", "Choose a short WAV or record a clip in your browser."),
        (b, "02", "Listen to features", "The app extracts MFCC, chroma, log-Mel and spectral summaries."),
        (c, "03", "Review", "See the predicted emotion, probabilities, waveform and evaluation metrics."),
    ]:
        with col:
            st.markdown(f'<div class="card"><span style="color:#2563eb;font-weight:800">{number}</span><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

elif page == "Analyze":
    st.markdown('<div class="page-head"><h2>Analyze audio</h2><p>Upload a WAV clip or record a short sample to see its predicted emotion.</p></div>', unsafe_allow_html=True)
    mode = st.radio("Audio source", ["Upload WAV", "Record voice"], horizontal=True)
    audio = None
    source = ""
    if mode == "Upload WAV":
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"])
        if uploaded is not None:
            audio = uploaded.getvalue()
            source = uploaded.name
            st.audio(audio, format="audio/wav")
    else:
        recording = st.audio_input("Record 3–5 seconds of speech")
        if recording is not None:
            audio = recording.getvalue()
            source = "Microphone recording"
            st.audio(audio, format="audio/wav")

    if audio:
        if st.button("Analyze clip", type="primary", disabled=bundle is None):
            try:
                with st.spinner("Analyzing audio…"):
                    result = analyse_audio(audio, source, bundle, classes)
                st.session_state.last_result = result
                save_history(result)
            except Exception as exc:
                st.error(f"Could not process this audio: {exc}")
        result = st.session_state.last_result
        if result and result["digest"] == hashlib.sha256(audio).hexdigest():
            left, right = st.columns([1, 1.2], gap="large")
            with left:
                color = COLORS.get(result["emotion"], "#2563eb")
                st.markdown(f'<div class="result" style="background:linear-gradient(135deg,{color},{color}cc)"><div class="eyebrow">Detected emotion</div><div class="emotion">{result["emotion"].title()}</div><div class="confidence">Confidence {result["confidence"]:.0%}</div></div>', unsafe_allow_html=True)
                if result["confidence"] < .5:
                    st.warning("Low confidence: the model is uncertain about this clip.")
                if result["duration"] > 10:
                    st.info("The clip is longer than the short acted sentences used in training.")
                display_probabilities(result["probabilities"], classes, result["emotion"])
            with right:
                st.pyplot(result["waveform"], clear_figure=True)
                plt.close(result["waveform"])
                st.caption(f"Duration: {result['duration']:.1f} seconds · Source: {result['source']}")
    else:
        st.markdown('<div class="empty">Your prediction will appear here after you upload or record a clip.</div>', unsafe_allow_html=True)

elif page == "History":
    st.markdown('<div class="page-head"><h2>Session history</h2><p>Recent predictions are kept in this browser session only.</p></div>', unsafe_allow_html=True)
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Analyze a clip and it will appear here.")
    else:
        frame = pd.DataFrame(history)
        avg = np.mean([float(x["Confidence"].rstrip("%")) for x in history])
        cols = st.columns(3)
        with cols[0]: kpi(str(len(history)), "Predictions")
        with cols[1]: kpi(frame["Emotion"].mode().iloc[0], "Most predicted")
        with cols[2]: kpi(f"{avg:.0f}%", "Average confidence")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Download history CSV", frame.to_csv(index=False), "prediction_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.seen = set()
            st.rerun()

elif page == "Dashboard":
    st.markdown('<div class="page-head"><h2>Model dashboard</h2><p>Evaluation on actors kept out of training and tuning.</p></div>', unsafe_allow_html=True)
    if not metrics or metrics.get("_load_error"):
        st.info(f"Add {METRICS_PATH.name} beside app.py to show test results.")
    else:
        cols = st.columns(4)
        kpis = [
            (f"{metrics.get('test_accuracy', 0):.1%}", "Accuracy"),
            (f"{metrics.get('test_macro_f1', 0):.3f}", "Macro F1"),
            (str(metrics.get("test_clips", "—")), "Test clips"),
            (metrics.get("selected_model", "—"), "Selected model"),
        ]
        for col, (value, label) in zip(cols, kpis):
            with col: kpi(value, label)
        st.caption("Held-out actors: " + ", ".join(map(str, metrics.get("test_actors", []))))
        left, right = st.columns(2, gap="large")
        counts = metrics.get("dataset_class_counts", {})
        if counts:
            with left:
                st.markdown('<div class="section">Dataset class counts</div>', unsafe_allow_html=True)
                st.bar_chart(pd.Series(counts, name="Clips"))
        confusion = metrics.get("confusion_matrix")
        if confusion:
            with right:
                st.markdown('<div class="section">Confusion matrix</div>', unsafe_allow_html=True)
                fig = confusion_figure(confusion)
                if fig is not None:
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
        report = metrics.get("classification_report", {})
        per_class = {name.title(): report[name] for name in classes if name in report and isinstance(report[name], dict)}
        if per_class:
            st.markdown('<div class="section">Per-emotion performance</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(per_class).T, use_container_width=True)
        st.caption(f"Macro F1 gives each emotion equal weight. Test accuracy: {metrics.get('test_accuracy', 0):.3f}.")

else:
    st.markdown('<div class="page-head"><h2>About VoxSense</h2><p>A small actor-independent speech emotion recognition demo.</p></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><h4>Dataset</h4><p>RAVDESS acted speech, filtered to angry, happy, sad and neutral. The final evaluation holds actors 19–24 out of training.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>Features and model</h4><p>MFCC and delta-MFCC, chroma, log-Mel and spectral summaries feed a class-balanced scikit-learn classifier.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h4>Limitations</h4><p>The model learned from acted studio speech. Real calls, background noise, accents and languages outside the training data may change results.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>Audio privacy</h4><p>Uploaded audio is processed in memory for a prediction. Session history stores prediction summaries only; it does not save the audio.</p></div>', unsafe_allow_html=True)
