# VoxSense — RAVDESS Speech Emotion AI

Streamlit demo for the four-class RAVDESS speech-emotion classifier. It follows the visual direction of the supplied example: dark branded sidebar, Home / Analyze / History / Dashboard / About navigation, audio upload or microphone recording, waveform, probability bars, and evaluation metrics.

## Fix for the missing features module

This version contains the exact feature-extraction function inside app.py. It does not import features.py, so the error ModuleNotFoundError: No module named 'features' is avoided. Replace the old app.py in your GitHub repository with this one, commit the change, and redeploy the Streamlit app.

## Files to download from Kaggle

Run the latest Kaggle notebook from top to bottom. In its Output panel, download:

- ravdess_emotion_model.joblib — required for audio predictions.
- ravdess_test_metrics.json — required for the Dashboard page's class counts, held-out accuracy/macro-F1, and confusion matrix.

The feature table ravdess_audio_features.csv is optional and is not needed by this app. Download links can be printed in the notebook with:

    from IPython.display import FileLink, display
    display(FileLink('/kaggle/working/ravdess_emotion_model.joblib'))
    display(FileLink('/kaggle/working/ravdess_test_metrics.json'))

Use the same model export and notebook version together: their feature extraction and feature order must match.

## Add files and run

Keep app.py, requirements.txt, ravdess_emotion_model.joblib, and ravdess_test_metrics.json in the repository root (the two Kaggle exports are not included in this source file set). Install dependencies with:

    pip install -r requirements.txt

Run the dashboard with:

    streamlit run app.py

The model is loaded from the repository root. Optional environment variables MODEL_PATH and METRICS_PATH can point to different locations. If scikit-learn displays a version warning, match the version stored in the model bundle. Only load joblib files generated from a source you trust.

## Deploy with Streamlit Community Cloud

1. Push the updated app.py and requirements.txt to GitHub.
2. Add ravdess_emotion_model.joblib and ravdess_test_metrics.json to the app's repository root (or configure secure paths in your deployment).
3. In Streamlit Community Cloud, choose the repository, branch, and app.py as the main file, then deploy.

The notebook uses actors 1–18 for model selection/training and actors 19–24 for final testing. Report the actual metrics exported by your run; the app does not invent performance numbers.
