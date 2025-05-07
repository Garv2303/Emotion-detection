import gradio as gr
import joblib
import os
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Helper Functions ---

# Load model and vectorizer
def load_model():
    return joblib.load("emotion_model.joblib")

# Clean input text
def clean_text(text):
    text = nfx.remove_punctuations(text)
    text = nfx.remove_stopwords(text)
    return text.lower()

# Predict emotion
def predict(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)
    return prediction[0]

# Save correction and retrain model
def save_correction(text, correct_label):
    # Save to CSV
    df = pd.DataFrame([[text, correct_label]], columns=["text", "emotion"])
    if os.path.exists("corrections.csv"):
        df.to_csv("corrections.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("corrections.csv", index=False)

    # Retrain model with updated data
    retrain_model()

# Retrain using original + corrections
def retrain_model():
    # Load original data
    texts, labels = load_data("train.txt")

    # Load corrections if available
    if os.path.exists("corrections.csv"):
        correction_df = pd.read_csv("corrections.csv")
        texts += correction_df["text"].tolist()
        labels += correction_df["emotion"].tolist()

    # Clean text
    texts_clean = [clean_text(t) for t in texts]

    # Vectorize
    X = vectorizer.fit_transform(texts_clean)

    # Retrain
    model.fit(X, labels)

    # Save updated model
    joblib.dump((vectorizer, model), "emotion_model.joblib")

# Load data from text file
def load_data(path):
    texts = []
    emotions = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) == 2:
                text, emotion = parts
                texts.append(text)
                emotions.append(emotion)
    return texts, emotions

# --- Initialize ---
vectorizer, model = load_model()

# --- Gradio Interface Logic ---

def full_pipeline(user_input, feedback, corrected_emotion):
    predicted = predict(user_input)
    
    # If feedback is no and correction provided, save correction
    if feedback == "No" and corrected_emotion:
        save_correction(user_input, corrected_emotion)
        message = f"Thanks! The model has been updated with your correction: {corrected_emotion}"
    elif feedback == "Yes":
        message = "Great! Thanks for confirming the prediction."
    else:
        message = "Please provide the correct emotion if the prediction was wrong."
    
    return predicted, message

# Interface layout
demo = gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Textbox(label="Enter a sentence"),
        gr.Radio(["Yes", "No"], label="Was this prediction correct?", value="Yes"),
        gr.Textbox(label="If 'No', enter the correct emotion")
    ],
    outputs=[
        gr.Textbox(label="Predicted Emotion"),
        gr.Textbox(label="Feedback Message")
    ],
    title="Emotion Detection with Feedback Loop",
    description="The model learns from its mistakes. Enter text, review the prediction, and help improve it!"
)

demo.launch()
