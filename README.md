AnnRakshak AI Assistant
Project Overview
AnnRakshak is an AI-powered assistant that helps users predict the shelf life of crops based on current weather conditions in their city. It also generates spoilage risk graphs and provides follow-up Q&A for better crop storage and handling.

Features Implemented
Voice and text input support

Automatic language detection (English or Hindi)

Translation support for Hindi to English and vice versa

Crop name extraction and fuzzy matching

Weather data fetching using OpenWeatherMap API

Dynamic shelf life prediction based on temperature

Spoilage risk graph generation using matplotlib

Smart AI-generated Q&A suggestions for the crop

AI-powered answers on question click using Gemini

Error handling for unknown crops or weather issues

Custom Gradio-based web interface with voice button

Secure API key management using .env file

Technologies Used
Python

Gradio

Google Gemini API

OpenWeatherMap API

Deep Translator

Speech Recognition

Matplotlib

Pandas, NumPy

How to Run
Clone the repository

Place your dataset enriched_crop_dataset.csv in the project root

Create a .env file in the same directory and add:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_api_key
WEATHER_API_KEY=your_openweather_api_key
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Run the app:

nginx
Copy
Edit
python your_script_name.py