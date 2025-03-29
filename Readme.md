# Pix2Pix Interactive Web App

## Overview
This is an interactive web application that allows users to generate realistic images from sketches using a Pix2Pix deep learning model. The app is built using Flask for the backend and JavaScript for the frontend, enabling real-time sketch-to-image translation.
(score is pretty low.. because of limited resources)


<video controls src="background/sketch_image.mp4" title="Title"></video>


## Features
- **Sketch Upload:** Users can upload their sketches via drag-and-drop or file input.
- **Model Selection:** Choose between different Pix2Pix models trained on various datasets.
- **Real-time Processing:** The app processes the sketch and returns a generated image.
- **Dark/Light Theme Toggle:** Switch between UI themes for a better user experience.
- **Download Generated Images:** Save the output images directly to your device.

## Installation
### Prerequisites
- Python 3.8+
- Flask
- PyTorch
- OpenCV
- JavaScript-enabled browser

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Pix2Pix-WebApp.git
   cd Pix2Pix-WebApp
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```sh
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in your browser.



## API Endpoints
- `POST /generate` - Uploads a sketch and generates an image.
- `GET /download/png/<filename>` - Downloads the PNG version of the generated image.
- `GET /download/jpg/<filename>` - Downloads the JPG version of the generated image.

## Technologies Used
- Python (Flask, OpenCV, NumPy)
- JavaScript (Frontend logic, theme toggle, form handling)
- HTML/CSS (UI/UX Design)
- Deep Learning (Pix2Pix, GAN models)


## Usage
1. Upload a sketch using the provided UI.
2. Select a model (e.g., `flower`, `shoes`, etc.).
3. Click **Generate** to process the sketch.
4. Download the generated image if satisfied.

## Folder Structure
```
Pix2Pix-WebApp/
│── static/
│   ├── css/          # Stylesheets
│   ├── js/           # Frontend JavaScript
│   ├── images/       # Default images and assets
│── templates/
│   ├── index.html    # Webpage template
│── models/           # Pre-trained Pix2Pix models
│── app.py            # Flask backend
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
│── .gitignore        # Git ignore file
```

## Deployment
- You can deploy the Flask app on **Render**, **Hugging Face Spaces**, or **Heroku**.
- The pre-trained models can be hosted on **Hugging Face Model Hub**.



