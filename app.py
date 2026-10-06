import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_features

BASE = Path(__file__).parent
COLOR = {"angry": "#ef4444", "happy": "#f59e0b", "neutral": "#71717a", "sad": "#0284c7"}
CUES = {
    "angry": "Loud, tense and sharp energy",
    "happy": "Bright, lively and higher pitched",
    "neutral": "Calm, even and steady tone",
    "sad": "Quiet, low energy and slower",
}
GRAY = "#71717a"
PAGES = ["Home", "Analyze", "History", "Dashboard", "About"]

st.set_page_config(page_title="VoxSense - Speech Emotion AI", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {background: #fafafa;}
.block-container {padding: 3.4rem 2.6rem 3rem; max-width: 1480px;}
h1, h2, h3, h4 {font-family: 'Manrope', 'Inter', sans-serif; color: #18181b; letter-spacing: -.01em;}

[data-testid="stSidebar"] {background: linear-gradient(180deg, #09090b 0%, #18181b 100%); border-right: 1px solid #27272a;}
[data-testid="stSidebar"] * {color: #e4e4e7;}
[data-testid="stSidebar"] div[role="radiogroup"] {gap: .15rem;}
[data-testid="stSidebar"] div[role="radiogroup"] label {padding: .55rem .8rem; border-radius: 10px; width: 100%; transition: background .15s;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: #27272a;}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: #2563eb;}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {color: #fff; font-weight: 600;}
.brand {display: flex; align-items: center; gap: .75rem; padding: .4rem .2rem 1rem;}
.logo {width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #3b82f6, #1d4ed8);
       display: flex; align-items: center; justify-content: center; gap: 3px;}
.logo i {display: block; width: 4px; border-radius: 2px; background: #fff;}
.brand b {display: block; font-family: 'Manrope', sans-serif; font-size: 1.15rem; color: #fff;}
.brand span {font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: #a1a1aa;}
.side-card {margin-top: 1.4rem; padding: 1rem; border-radius: 14px; background: #18181b; border: 1px solid #27272a;}
.side-card .lbl {font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: #a1a1aa;}
.side-card .big {font-family: 'Manrope', sans-serif; font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1.2;}
.side-row {display: flex; justify-content: space-between; margin-top: .6rem; padding-top: .6rem; border-top: 1px solid #27272a; font-size: .8rem;}
.side-row b {color: #fff;}

.ph h2 {margin: 0 0 .15rem; font-size: 1.7rem;}
.ph p {margin: 0 0 1.2rem; color: #71717a;}

.hero {position: relative; overflow: hidden; border-radius: 20px; padding: 3rem 2.6rem; color: #fff; margin-bottom: 1.3rem;
       background: radial-gradient(circle at 85% 20%, rgba(59,130,246,.55) 0%, rgba(59,130,246,0) 45%),
                   linear-gradient(135deg, #09090b 0%, #18181b 55%, #1e3a8a 130%);}
.hero .tag {display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-size: .72rem; font-weight: 600;
            letter-spacing: .1em; text-transform: uppercase; background: rgba(59,130,246,.2); border: 1px solid rgba(96,165,250,.5); color: #bfdbfe;}
.hero h1 {color: #fff; font-size: 2.7rem; line-height: 1.1; margin: 1rem 0 .7rem; padding: 0;}
.hero p {max-width: 640px; color: #d4d4d8; font-size: 1.05rem; margin: 0;}
.pill {display: inline-block; margin: 1.3rem .4rem 0 0; padding: .25rem .85rem; border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff;}

.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: .4rem 0 1.4rem;}
.kpi {background: #fff; border: 1px solid #e4e4e7; border-top: 3px solid #2563eb; border-radius: 14px; padding: 1.1rem 1.2rem;
      box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.kpi .v {font-family: 'Manrope', sans-serif; font-size: 1.95rem; font-weight: 800; color: #18181b; line-height: 1.15;}
.kpi .k {font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: #71717a; margin-top: .2rem;}
.card {background: #fff; border: 1px solid #e4e4e7; border-radius: 14px; padding: 1.2rem 1.35rem; margin-bottom: .9rem;
       box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.card h4 {margin: 0 0 .5rem; font-size: 1.02rem;}
.card p {margin: 0; color: #52525b; font-size: .93rem; line-height: 1.55;}
.num {display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 8px;
      background: #eff6ff; color: #2563eb; font-weight: 700; font-size: .85rem; margin-bottom: .6rem;}
.dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .5rem;}
.section {font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 800; color: #18181b; margin: 1.2rem 0 .6rem;}
.tip {border-left: 4px solid #2563eb; padding: .7rem 1rem; border-radius: 8px; background: #eff6ff;
      color: #1e3a8a; margin-bottom: 1rem; font-size: .92rem;}
.placeholder {border: 2px dashed #d4d4d8; border-radius: 16px; padding: 3.2rem 1.5rem; text-align: center; color: #71717a; background: #fff;}
.placeholder b {display: block; color: #18181b; font-size: 1.05rem; margin-bottom: .3rem;}

.result {border-radius: 16px; padding: 1.6rem 1.5rem; color: #fff; margin-bottom: .9rem;}
.result .k {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; opacity: .85;}
.result .label {font-family: 'Manrope', sans-serif; font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: .2rem 0 .9rem; color: #fff;}
.result .meter {height: 8px; border-radius: 999px; background: rgba(255,255,255,.3); overflow: hidden;}
.result .meter div {height: 100%; background: #fff; border-radius: 999px;}
.result .conf {margin-top: .5rem; font-size: .93rem; opacity: .95;}
.bar-row {display: flex; align-items: center; margin: .6rem 0; gap: .8rem;}
.bar-name {width: 90px; font-weight: 600; color: #27272a;}
.bar-track {flex: 1; height: 12px; border-radius: 999px; background: #e4e4e7; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-val {width: 48px; text-align: right; font-variant-numeric: tabular-nums; color: #3f3f46;}

@media (max-width: 768px) {
  .block-container {padding: 3rem 1rem 2rem;}
  .hero {padding: 1.8rem 1.3rem;}
  .hero h1 {font-size: 1.9rem;}
  .result .label {font-size: 2rem;}
  .bar-name {width: 72px;}
}
</style>
""",
    unsafe_allow_html=True,
)

for key, default in [("history", []), ("seen", set()), ("rec_n", 0), ("up_n", 0), ("page", "Home")]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_metrics():
    with open(BASE / "metrics.json") as f:
        return json.load(f)


def predict_proba(file_bytes, model, feature_idx):
    vec = extract_features(io.BytesIO(file_bytes))[feature_idx]
    return pd.Series(model.predict_proba(vec.reshape(1, -1))[0], index=model.classes_)


def page_header(title, subtitle):
    st.markdown(f'<div class="ph"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_grid(items):
    html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in items)
    st.markdown(f'<div class="grid">{html}</div>', unsafe_allow_html=True)


def bars_html(values, classes, maximum=1.0, as_percent=True, highlight=None):
    rows = []
    for c in classes:
        v = float(values[c])
        width = 100 * v / maximum if maximum else 0
        label = f"{v:.0%}" if as_percent else f"{int(v)}"
        opacity = 1 if (highlight is None or c == highlight) else 0.4
        rows.append(
            f'<div class="bar-row"><div class="bar-name">{c.capitalize()}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{COLOR[c]};opacity:{opacity}"></div></div>'
            f'<div class="bar-val">{label}</div></div>'
        )
    return "".join(rows)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d4d4d8")
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.title.set_color(GRAY)


def plot_waveform(file_bytes, sr, color):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr)
    fig, ax = plt.subplots(figsize=(11, 2.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    style_axes(ax)
    fig.tight_layout()
    return fig, len(y) / sr


def plot_confusion(cm, classes):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_alpha(0)
    names = [c.capitalize() for c in classes]
    for ax, data, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised", ".2f")]:
        ax.patch.set_alpha(0)
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes)), names)
        ax.set_yticks(range(len(classes)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        style_axes(ax)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "#111", fontweight="bold")
    fig.tight_layout()
    return fig


def add_to_history(file_bytes, source, proba):
    digest = hashlib.md5(file_bytes).hexdigest()
    if digest in st.session_state.seen:
        return
    st.session_state.seen.add(digest)
    row = {"Time": datetime.now().strftime("%H:%M:%S"), "Source": source,
           "Emotion": proba.idxmax().capitalize(), "Confidence": f"{proba.max():.0%}"}
    for c in proba.index:
        row[c.capitalize()] = f"{proba[c]:.0%}"
    st.session_state.history.insert(0, row)


def analyze(file_bytes, source, model, metrics):
    try:
        with st.spinner("Analysing audio..."):
            proba = predict_proba(file_bytes, model, metrics["feature_indices"])
            top = proba.idxmax()
            fig, duration = plot_waveform(file_bytes, metrics["sample_rate"], COLOR[top])
    except Exception as e:
        st.error(f"Could not process this audio. Please try another recording. ({e})")
        return None
    add_to_history(file_bytes, source, proba)
    return {"proba": proba, "top": top, "fig": fig, "duration": duration}


def render_result(res, classes):
    proba, top = res["proba"], res["top"]
    st.markdown(
        f'<div class="result" style="background:linear-gradient(135deg,{COLOR[top]},{COLOR[top]}cc)">'
        f'<div class="k">Detected emotion</div><div class="label">{top.capitalize()}</div>'
        f'<div class="meter"><div style="width:{proba.max()*100:.0f}%"></div></div>'
        f'<div class="conf">Confidence {proba.max():.0%}</div></div>',
        unsafe_allow_html=True,
    )
    if proba.max() < 0.5:
        st.warning("Low confidence: the model is unsure about this clip.")
    if res["duration"] > 10:
        st.warning("This clip is long. The model was trained on clips of about 3-4 seconds.")
    if res["duration"] < 1:
        st.warning("This clip is very short. Try speaking for 3-5 seconds.")
    st.markdown(
        f'<div class="card"><h4>Probability per emotion</h4>{bars_html(proba, classes, highlight=top)}</div>',
        unsafe_allow_html=True,
    )


def new_recording():
    st.session_state.rec_n += 1


def new_upload():
    st.session_state.up_n += 1


def go_analyze():
    st.session_state.page = "Analyze"


model = load_model()
metrics = load_metrics()
classes = metrics["classes"]

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo"><i style="height:10px"></i><i style="height:20px"></i><i style="height:14px"></i>'
        '<i style="height:24px"></i><i style="height:12px"></i></div>'
        '<div><b>VoxSense</b><span>Speech emotion AI</span></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.markdown(
        f'<div class="side-card"><div class="lbl">Model accuracy</div><div class="big">{metrics["accuracy"]:.1%}</div>'
        f'<div class="side-row"><span>Macro F1</span><b>{metrics["macro_f1"]:.3f}</b></div>'
        f'<div class="side-row"><span>Test clips</span><b>{metrics["n_test"]}</b></div>'
        f'<div class="side-row"><span>Predictions today</span><b>{len(st.session_state.history)}</b></div></div>',
        unsafe_allow_html=True,
    )

if page == "Home":
    pills = "".join(f'<span class="pill" style="background:{COLOR[c]}">{c.capitalize()}</span>' for c in classes)
    st.markdown(
        '<div class="hero"><span class="tag">AI powered speech analysis</span>'
        '<h1>Hear the emotion behind every voice</h1>'
        '<p>Record your voice or upload a short clip. The model listens to the tone of the speech and tells you whether it sounds '
        f'angry, happy, sad or neutral, in under a second.</p>{pills}</div>',
        unsafe_allow_html=True,
    )
    st.button("Start analysing", type="primary", on_click=go_analyze)
    kpi_grid([
        (f"{metrics['accuracy']:.1%}", "Test accuracy"),
        (f"{metrics['macro_f1']:.3f}", "Macro F1"),
        ("4", "Emotions detected"),
        (f"{metrics['n_train_original'] + metrics['n_test']}", "Clips in dataset"),
    ])

    st.markdown('<div class="section">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Capture", "Record with your microphone or upload a WAV file. Silence at the start and end is trimmed automatically."),
        ("2", "Extract features", "Each clip becomes 213 numbers that describe the voice: MFCC, chroma, mel energy, spectral contrast, loudness and brightness."),
        ("3", "Predict", f"A {metrics['model']} classifier with balanced class weights returns a probability for every emotion."),
    ]
    html = "".join(f'<div class="card"><div class="num">{n}</div><h4>{t}</h4><p>{d}</p></div>' for n, t, d in steps)
    st.markdown(f'<div class="grid">{html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">What the model listens for</div>', unsafe_allow_html=True)
    html = "".join(
        f'<div class="card"><h4><span class="dot" style="background:{COLOR[c]}"></span>{c.capitalize()}</h4><p>{CUES[c]}</p></div>'
        for c in classes
    )
    st.markdown(f'<div class="grid">{html}</div>', unsafe_allow_html=True)

elif page == "Analyze":
    page_header("Analyze audio", "Record your voice or upload a clip to detect the emotion.")
    mode = st.radio("Input method", ["Record my voice", "Upload an audio file"], horizontal=True, key="mode")

    left, right = st.columns([1, 1.15], gap="large")
    audio_bytes, source = None, None
    with left:
        st.markdown(
            '<div class="tip"><b>Tip:</b> the model was trained on acted speech. Speak for 3-5 seconds in a quiet room and '
            'exaggerate the emotion, for example say <i>"Kids are talking by the door"</i> in an angry, happy or sad voice.</div>',
            unsafe_allow_html=True,
        )
        if mode == "Record my voice":
            rec = st.audio_input("Press the microphone, speak, then press stop", key=f"rec_{st.session_state.rec_n}")
            if rec is None:
                st.info("Allow microphone access in your browser, then press the microphone button and speak.")
            else:
                audio_bytes, source = rec.getvalue(), "Microphone"
                st.button("Record a new clip", on_click=new_recording, key="again_rec")
        else:
            up = st.file_uploader("Upload a WAV file", type=["wav"], key=f"up_{st.session_state.up_n}")
            if up is None:
                st.info("Choose a WAV file to see the predicted emotion, probabilities and waveform.")
            else:
                audio_bytes, source = up.getvalue(), f"Upload: {up.name}"
                st.audio(audio_bytes, format="audio/wav")
                st.button("Upload another file", on_click=new_upload, key="again_up")

    res = analyze(audio_bytes, source, model, metrics) if audio_bytes else None
    with right:
        if res:
            render_result(res, classes)
        else:
            st.markdown(
                '<div class="placeholder"><b>Your result will appear here</b>'
                'Provide some audio on the left to see the detected emotion and probabilities.</div>',
                unsafe_allow_html=True,
            )
    if res:
        st.markdown('<div class="section">Waveform</div>', unsafe_allow_html=True)
        st.pyplot(res["fig"])
        st.caption(f"Duration after trimming silence: {res['duration']:.1f} s. Saved to the History page.")

elif page == "History":
    page_header("Prediction history", "Every prediction you make in this session is saved here.")
    hist = st.session_state.history
    if not hist:
        st.info("No predictions yet. Record or upload audio on the Analyze page and it will appear here.")
    else:
        hdf = pd.DataFrame(hist)
        counts = hdf["Emotion"].str.lower().value_counts().reindex(classes).fillna(0)
        avg_conf = hdf["Confidence"].str.rstrip("%").astype(float).mean()
        kpi_grid([
            (str(len(hdf)), "Predictions"),
            (hdf["Emotion"].mode()[0], "Most frequent emotion"),
            (f"{avg_conf:.0f}%", "Average confidence"),
        ])
        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            st.markdown(
                f'<div class="card"><h4>Predicted emotions</h4>'
                f'{bars_html(counts, classes, maximum=max(counts.max(), 1), as_percent=False)}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.dataframe(hdf, hide_index=True)
        b1, b2, _ = st.columns([1, 1, 4])
        b1.download_button("Download CSV", hdf.to_csv(index=False), "prediction_history.csv", "text/csv")
        if b2.button("Clear history"):
            st.session_state.history = []
            st.session_state.seen = set()
            st.rerun()
        st.caption("History is kept for the current browser session only.")

elif page == "Dashboard":
    page_header("Model dashboard", "Performance on actors the model has never heard.")
    kpi_grid([
        (f"{metrics['accuracy']:.1%}", "Accuracy"),
        (f"{metrics['macro_f1']:.3f}", "Macro F1"),
        (str(metrics["n_test"]), "Test clips"),
        (str(metrics["n_train_total"]), "Training rows"),
    ])
    st.caption(
        f"Model: {metrics['model']} | Features: {metrics['feature_set']} | Training: {metrics['training_regime']} | "
        f"Train actors {min(metrics['train_actors'])}-{max(metrics['train_actors'])}, "
        f"test actors {min(metrics['test_actors'])}-{max(metrics['test_actors'])}"
    )
    left, right = st.columns(2, gap="large")
    with left:
        cc = pd.Series(metrics["class_counts_all"])
        st.markdown(
            f'<div class="card"><h4>Class counts (672 clips)</h4>'
            f'{bars_html(cc, classes, maximum=cc.max(), as_percent=False)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        report = metrics["per_class_report"]
        table = pd.DataFrame(
            {c.capitalize(): {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes}
        ).T
        st.markdown('<div class="section" style="margin-top:0">Per-class scores</div>', unsafe_allow_html=True)
        st.dataframe(table.round(3))

    st.markdown('<div class="section">Confusion matrix</div>', unsafe_allow_html=True)
    st.pyplot(plot_confusion(metrics["confusion_matrix"], classes))
    with st.expander("All experiments (cross-validation vs test)"):
        st.dataframe(pd.DataFrame(metrics["experiments"]).round(3))

else:
    page_header("About this project", "Speech emotion detection on the RAVDESS dataset.")
    items = [
        ("Data", "RAVDESS speech audio with 4 emotions (angry, happy, sad, neutral): 672 clips from 24 professional actors. Neutral has 96 clips, the others 192 each."),
        ("Features", "MFCC, delta-MFCC, chroma, log-mel, spectral contrast and loudness or brightness statistics, 213 numbers per clip."),
        ("Model", "A scikit-learn classifier with balanced class weights, tuned with actor-grouped cross-validation."),
        ("Evaluation", "Actors are split between training and testing, so test voices were never seen in training. This avoids data leakage."),
    ]
    html = "".join(f'<div class="card"><h4>{t}</h4><p>{d}</p></div>' for t, d in items)
    st.markdown(f'<div class="grid">{html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">Frequently asked questions</div>', unsafe_allow_html=True)
    with st.expander("Why is my result not always correct?"):
        st.write("The model learned from acted speech recorded in a studio. Real-life speech, background noise, accents and "
                 "other languages are harder. Speak clearly and exaggerate the emotion for the best result.")
    with st.expander("Is my voice stored?"):
        st.write("No. Audio is processed in memory to make a prediction. Only the prediction summary is kept in the History "
                 "page for your current browser session.")
    with st.expander("How long should the recording be?"):
        st.write("About 3 to 5 seconds. The training clips were short sentences of 3 to 4 seconds.")
    with st.expander("What does macro F1 mean?"):
        st.write("It is the average F1 score over all emotions, so the smaller neutral class counts as much as the others.")
