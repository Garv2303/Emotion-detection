# 🧠 Emotion Detection from Text (with Feedback Loop)

This project is a beginner-friendly emotion detection system that uses machine learning to identify the emotion (like joy, sadness, anger, etc.) in text inputs. It features a feedback loop where users can correct wrong predictions, and the model learns from those corrections in real-time.

## ✅ Features

- Detect emotions from user-entered sentences
- Works entirely offline (no API or internet required)
- Asks for user feedback after each prediction
- Automatically updates and retrains the model based on feedback
- Corrections saved in `corrections.csv` for future learning
- Built using: `scikit-learn`, `joblib`, `neattext`, `numpy`, `seaborn`.
