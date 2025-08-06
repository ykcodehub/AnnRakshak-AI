import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gradio as gr
from deep_translator import GoogleTranslator
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from difflib import get_close_matches
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()

# ========== API KEYS ==========
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-pro")

# ========== DATA ==========
df = pd.read_csv("enriched_crop_dataset.csv")

# ========== WEATHER API ==========
def get_weather(city, api_key=WEATHER_API_KEY):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        return temp, humidity
    except Exception:
        return None, None

# ========== CROP NAME MATCH ==========
def match_crop_name(input_crop):
    input_crop = input_crop.lower().strip()
    matched = df['crop'].str.lower().str.strip() == input_crop
    if matched.any():
        return df[matched].iloc[0]['crop']
    crops = df['crop'].str.lower().unique()
    for crop in crops:
        if input_crop in crop or crop in input_crop:
            return crop
    return None

# ========== SHELF LIFE PREDICTION ==========
def get_crop_insight(crop, temp):
    crop = match_crop_name(crop)
    if crop is None:
        return {"error": "माफ़ कीजिए, हमारे पास इस फसल से संबंधित डेटा उपलब्ध नहीं है।"}
    row = df[df['crop'].str.lower() == crop.lower()].iloc[0]

    def safe_int(value):
        try:
            return int(float(value))
        except:
            return 0

    min_temp = float(row['min_temp'])
    max_temp = float(row['max_temp'])

    if temp < min_temp:
        shelf_life = safe_int(row['shelf_life_low_temp'])
        temp_range = "Low"
    elif temp > max_temp:
        shelf_life = safe_int(row['shelf_life_high_temp'])
        temp_range = "High"
    else:
        shelf_life = safe_int(row['shelf_life_mid_temp'])
        temp_range = "Ideal"

    return {
        "crop": crop,
        "temperature": temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "estimated_shelf_life_days": shelf_life,
        "temp_range": temp_range,
        "storage_tips": row['storage_tips']
    }

# ========== TRANSLATION ==========
def detect_language(text):
    hindi_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
    return "hi" if any(char in hindi_chars for char in text) else "en"

def translate_to_english(text):
    lang = detect_language(text)
    return (GoogleTranslator(source='auto', target='en').translate(text), lang) if lang == "hi" else (text, lang)

# ========== AI RESPONSE ==========
def generate_crop_response(crop_input, location, lang):
    matched_crop = match_crop_name(crop_input)
    if not matched_crop:
        return "माफ़ कीजिए, हमारे पास इस फसल से संबंधित डेटा उपलब्ध नहीं है।"
    temp, humidity = get_weather(location)
    if temp is None:
        return "Weather data could not be fetched."
    crop_data = get_crop_insight(matched_crop, temp)
    if "error" in crop_data:
        return crop_data["error"]
    response = f"""
Crop: {matched_crop.title()}
Temperature: {temp}°C ({crop_data['temp_range']} range)
Humidity: {humidity}%
Estimated Shelf Life: {crop_data['estimated_shelf_life_days']} days
Tips: {crop_data['storage_tips']}"""
    return GoogleTranslator(source='en', target='hi').translate(response) if lang == "hi" else response

# ========== GRAPH OUTPUT ==========
def plot_spoilage_risk_graph(crop, temperature, humidity):
    temp_range = np.arange(0, 50, 1)
    risk_values = [2 if t<=10 else 4 if t<=20 else 6 if t<=30 else 8+(t-30)*0.5 for t in temp_range]
    plt.figure(figsize=(10, 6))
    plt.plot(temp_range, risk_values, color='black', linestyle='-', linewidth=2, label="Spoilage Risk Curve")
    plt.axvspan(0, 10, facecolor='green', alpha=0.3, label="Better Temp (0-10°C)")
    plt.axvspan(10, 20, facecolor='yellow', alpha=0.3, label="Normal (10-20°C)")
    plt.axvspan(20, 30, facecolor='orange', alpha=0.3, label="Moderate (20-30°C)")
    plt.axvspan(30, 50, facecolor='red', alpha=0.3, label="Risk (>30°C)")
    plt.axvline(x=temperature, color='blue', linestyle='--', label=f"Your Temp: {temperature}°C")
    plt.title(f"Spoilage Risk for {crop.title()} (Humidity: {humidity}%)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Spoilage Risk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = "spoilage_graph.png"
    plt.savefig(filename)
    plt.close()
    return filename

# ========== Q&A ==========
def generate_static_questions(crop, lang):
    base = [
        f"What is the ideal temperature to store {crop}?",
        f"How long can {crop} last in high humidity?",
        f"When should I sell or donate {crop}?",
        f"What are common spoilage signs in {crop}?",
        f"How can I increase the shelf life of {crop}?"
    ]
    questions = base if lang == "en" else [GoogleTranslator(source='en', target='hi').translate(q) for q in base]
    return [(q, crop, lang) for q in questions]

def generate_dynamic_answer(question_crop_lang):
    question, crop, lang = question_crop_lang
    prompt = f"You're a smart crop advisor. Please answer this question clearly: {question}" if lang == "en" else f"आप एक विशेषज्ञ कृषि सलाहकार हैं। कृपया इस प्रश्न का स्पष्ट उत्तर दें: {question}"
    try:
        response = gemini_model.generate_content(prompt)
        return [(response.text.strip(), None)]
    except:
        return [("Answer not available.", None)]

# ========== EXTRACTION ==========
def extract_crop_and_location(text):
    text = text.lower().strip()
    words = text.split()
    crops = df['crop'].str.lower().unique()
    found_crop = None
    found_city = None
    for word in words:
        matches = get_close_matches(word, crops, n=1, cutoff=0.6)
        if matches:
            found_crop = matches[0]
        elif not found_city:
            found_city = word
    return found_crop, found_city if found_city else "Delhi"

# ========== VOICE ==========
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="hi-IN")
    except:
        return ""

def main_function(user_text, language_selection):
    if user_text == "":
        user_text = get_voice_input()
    translated, lang = translate_to_english(user_text)
    crop, location = extract_crop_and_location(translated)
    if not crop:
        return "माफ़ कीजिए, हमारे पास इस फसल से संबंधित डेटा उपलब्ध नहीं है।", None, []
    response = generate_crop_response(crop, location, language_selection)
    temp, humidity = get_weather(location)
    graph_path = plot_spoilage_risk_graph(crop, temp, humidity) if temp else None
    qa_pairs = generate_static_questions(crop, language_selection)
    return response, graph_path, qa_pairs

def get_voice_and_run(lang):
    text = get_voice_input()
    return main_function(text, lang)

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
        <style>
        .mic-icon {
            background: url('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Google_mic.svg/2048px-Google_mic.svg.png') no-repeat center;
            background-size: contain;
            width: 32px;
            height: 32px;
            border: none;
            cursor: pointer;
            margin-top: 25px;
        }
        .theme-switch {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 32px;
            height: 32px;
            background: url('https://cdn-icons-png.flaticon.com/512/1828/1828774.png') no-repeat center;
            background-size: contain;
            cursor: pointer;
            border: none;
        }
        </style>
    """)

    gr.Markdown("""<h1 style='color:#38bdf8;'> AnnRakshak AI Assistant</h1>
    <p>Get shelf life, spoilage risk graph, and smart Q&A for crops based on city weather</p>""")

    with gr.Row():
        user_input = gr.Textbox(label="Enter Crop, Weather Info")
        mic_button = gr.Button(value="", elem_classes=["mic-icon"])
        lang_dropdown = gr.Dropdown(choices=["en", "hi"], value="en", label="Select Language")

    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")

    ai_output = gr.Textbox(label="AI Response")
    graph_plot = gr.Image(label="Spoilage Risk Graph")

    with gr.Accordion("Explore Follow-up Questions", open=True):
        qa_output = gr.HighlightedText(label="Dynamic Q&A")

    theme_button = gr.Button(value="", elem_classes=["theme-switch"])

    submit.click(fn=main_function, inputs=[user_input, lang_dropdown], outputs=[ai_output, graph_plot, qa_output])
    mic_button.click(fn=get_voice_and_run, inputs=[lang_dropdown], outputs=[ai_output, graph_plot, qa_output])
    clear.click(lambda: ("", None, []), outputs=[ai_output, graph_plot, qa_output])

    qa_output.select(fn=generate_dynamic_answer, inputs=[qa_output], outputs=qa_output)

app.launch()
