# RAVDESS Speech Emotion Detector

A Streamlit dashboard for the four-class audio-emotion model trained in the Kaggle notebook. It accepts WAV uploads, plots the waveform, predicts neutral/happy/sad/angry, and displays class probabilities. Add the notebook's test-metrics JSON to show dataset counts, held-out accuracy and macro-F1, per-emotion scores, and confusion matrix.

## Kaggle files to download

After running the final cells of the notebook, download these files from Kaggle's notebook Output panel:

- ravdess_emotion_model.joblib — required. Contains the fitted classifier and metadata used by this app.
- ravdess_test_metrics.json — recommended. Contains test accuracy, macro-F1, class counts, confusion matrix, and classification report.

ravdess_audio_features.csv is optional for inspecting the extracted feature table; this app does not need it. Do not download the original RAVDESS WAV dataset to run the dashboard; users upload a WAV when testing it.

The notebook stores outputs in /kaggle/working. To create download links inside Kaggle, run:

    from IPython.display import FileLink, display
    display(FileLink('/kaggle/working/ravdess_emotion_model.joblib'))
    display(FileLink('/kaggle/working/ravdess_test_metrics.json'))

Use the updated notebook that filters audio-only speech clips and saves the confusion matrix/class counts in the JSON. If the app has an older metrics JSON, rerun the notebook and download the new one.

## Run locally or in a Replit app

1. Put app.py, requirements.txt, ravdess_emotion_model.joblib, and ravdess_test_metrics.json in the same folder.
2. Install dependencies with: pip install -r requirements.txt
3. Start the dashboard with: streamlit run app.py

The model and metrics files are optional at startup, but predictions need the model; the evaluation page needs the metrics JSON. If scikit-learn shows a version warning, install the version recorded in the model bundle, then restart the app. Only load joblib files produced by a source you trust.

## Dataset and evaluation

RAVDESS speech files are filtered to audio-only speech and the four assessment emotions. Actors 1–18 are used for model selection/training and actors 19–24 are held out for final evaluation. The dashboard reports the actual values saved by the notebook; it does not invent or estimate scores.

## Project files

- app.py — Streamlit interface and matching librosa feature extraction.
- requirements.txt — Python dependencies.
- ravdess_emotion_model.joblib — add from Kaggle; not included in this source bundle.
- ravdess_test_metrics.json — add from Kaggle; not included in this source bundle.
