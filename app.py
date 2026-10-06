from __future__ import annotations

import io
import json
import os
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import sklearn
import streamlit as st

st.set_page_config(page_title="RAVDESS Emotion Detector", page_icon="🎙️", layout="wide")
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "ravdess_emotion_model.joblib"))
METRICS_PATH = Path(os.environ.get("METRICS_PATH", APP_DIR / "ravdess_test_metrics.json"))
FALLBACK_LABELS = ["neutral", "happy", "sad", "angry"]


def summarize_over_time(values: np.ndarray) -> np.ndarray:
    """Mean and standard deviation for each feature over time."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.concatenate([values.mean(axis=1), values.std(axis=1)])


def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Keep this preprocessing identical to the Kaggle notebook's feature cell."""
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))

    n_fft, hop = 2048, 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=64),
        ref=np.max,
    )
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spectral_summary = np.vstack([
        librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0],
    ])
    return np.concatenate([
        summarize_over_time(mfcc),
        summarize_over_time(delta_mfcc),
        summarize_over_time(chroma),
        summarize_over_time(mel_db),
        summarize_over_time(contrast),
        summarize_over_time(spectral_summary),
    ]).astype(np.float32)



@st.cache_resource
def load_model_bundle(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_metrics(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_uploaded_audio(audio_bytes: bytes, target_sr: int):
    y, original_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.size == 0:
        raise ValueError("The uploaded WAV contains no audio samples.")
    if original_sr != target_sr:
        y = librosa.resample(y, orig_sr=int(original_sr), target_sr=int(target_sr))
    return y.astype(np.float32), int(target_sr)


def show_evaluation(metrics):
    st.subheader("Held-out actor evaluation")
    if not metrics:
        st.info("Add ravdess_test_metrics.json beside app.py to show the notebook's evaluation results.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
    m2.metric("Test macro-F1", f"{metrics.get('test_macro_f1', 0):.3f}")
    m3.metric("Test clips", str(metrics.get("test_clips", "—")))
    st.caption(
        f"Model: {metrics.get('selected_model', '—')} · "
        f"Held-out actors: {', '.join(map(str, metrics.get('test_actors', []))) or '—'}"
    )

    counts = metrics.get("dataset_class_counts")
    if counts:
        st.markdown("**Dataset clips by emotion**")
        st.bar_chart(pd.Series(counts, name="clips"))

    confusion = metrics.get("confusion_matrix")
    if confusion and confusion.get("labels") and confusion.get("counts"):
        labels = confusion["labels"]
        matrix = pd.DataFrame(confusion["counts"], index=labels, columns=labels)
        st.markdown("**Confusion matrix — held-out actors**")
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows are true labels; columns are predicted labels.")

    report = metrics.get("classification_report")
    if report:
        rows = {name: values for name, values in report.items()
                if isinstance(values, dict) and name in FALLBACK_LABELS}
        if rows:
            st.markdown("**Per-emotion scores**")
            st.dataframe(pd.DataFrame(rows).T, use_container_width=True)


st.title("Speech Emotion Detector")
st.write("Upload a WAV clip to predict neutral, happy, sad, or angry. Evaluation metrics use actors kept out of model training.")
bundle = load_model_bundle(str(MODEL_PATH))
metrics = load_metrics(str(METRICS_PATH))

if bundle is None:
    st.warning(f"Model file not found: {MODEL_PATH.name}. Download it from Kaggle and place it beside app.py.")
else:
    trained_version = bundle.get("sklearn_version")
    if trained_version and trained_version != sklearn.__version__:
        st.warning(
            f"Model was saved with scikit-learn {trained_version}; this app has {sklearn.__version__}. "
            "For best compatibility, use the training version shown in the Kaggle model bundle."
        )

predict_tab, results_tab = st.tabs(["Try an audio clip", "Evaluation results"])
with predict_tab:
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        if bundle is not None:
            try:
                target_sr = int(bundle.get("sample_rate", 22_050))
                waveform, sr = read_uploaded_audio(audio_bytes, target_sr)
                fig, ax = plt.subplots(figsize=(11, 2.5))
                times = np.arange(len(waveform)) / sr
                ax.plot(times, waveform, linewidth=0.6, color="#3267a8")
                ax.set(title="Uploaded audio waveform", xlabel="Time (seconds)", ylabel="Amplitude")
                ax.grid(alpha=0.2)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                features = extract_audio_features(waveform, sr)
                expected_features = len(bundle.get("feature_columns", []))
                if expected_features and len(features) != expected_features:
                    raise ValueError(
                        f"Feature mismatch: this app made {len(features)} features, "
                        f"but the model expects {expected_features}. Use the matching notebook export."
                    )
                model = bundle["model"]
                X = features.reshape(1, -1)
                prediction = str(model.predict(X)[0])
                st.success(f"Predicted emotion: {prediction.title()}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                    class_names = (model.classes_ if hasattr(model, "classes_")
                                   else model.named_steps["classifier"].classes_)
                    probability_table = pd.DataFrame({
                        "Emotion": [str(name).title() for name in class_names],
                        "Probability": probabilities,
                    }).sort_values("Probability", ascending=False).set_index("Emotion")
                    st.markdown("**Class probabilities**")
                    st.bar_chart(probability_table)
                else:
                    st.info("This saved classifier does not provide predict_proba scores.")
            except Exception as exc:
                st.error(f"Could not process this WAV: {exc}")

with results_tab:
    show_evaluation(metrics)
