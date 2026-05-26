# 🖼️ AI Image Caption & Story Generator

An AI-powered web application that generates intelligent image captions from uploaded images and creates short creative stories based on those captions with optional user-provided keywords.

This project combines **computer vision, NLP, speech recognition, and text-to-speech** into a single interactive Streamlit application.

---

## ✨ Features

- 📸 Upload an image (JPG / JPEG / PNG)
- 🤖 Generate descriptive captions using **BLIP (Bootstrapping Language-Image Pretraining)**
- ✍️ Generate a short AI story based on the caption + optional keywords
- 🎤 Add keywords using **microphone speech input**
- 🔊 Convert generated stories into audio narration
- 🌐 Interactive **Streamlit web interface**
- 📓 Jupyter notebook included for experimentation and model training workflows

---

## 🛠 Tech Stack

### Frontend
- Streamlit

### AI / Machine Learning
- Hugging Face Transformers
- BLIP (Image Captioning)
- BART (Story Generation)

### Speech Processing
- SpeechRecognition
- Google Speech Recognition API
- gTTS (Google Text-to-Speech)

### Core Libraries
- Python
- TensorFlow / Keras
- Pillow
- NumPy

---

## 📂 Project Structure

```bash
ImageCaption_and_story_generator/
│
├── main.py                      # Streamlit web application
├── requirements.txt             # Python dependencies
├── captions.txt                 # Flickr8K captions dataset
├── image-caption.ipynb          # Training + experimentation notebook
├── models/
│   └── tokenizer.pkl            # Saved tokenizer
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/gunjan788/ImageCaption_and_story_generator.git
cd ImageCaption_and_story_generator
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal (usually):

```bash
http://localhost:8501
```

---

## 🎯 How It Works

1. User uploads an image
2. BLIP generates an image caption
3. User optionally provides keywords:
   - manually via text input
   - via microphone speech input
4. BART generates a short creative story
5. gTTS converts the story into speech

---

## 📚 Dataset

This project uses the **Flickr8K dataset**, which contains:

- 8,091 images
- 5 captions per image
- Natural language image descriptions

The notebook includes preprocessing, visualization, and experimentation with image-captioning pipelines.

---

## 📓 Jupyter Notebook

To explore training and experimentation:

```bash
jupyter notebook image-caption.ipynb
```

Notebook includes:

- Data preprocessing
- Visualization
- Feature extraction
- Model experimentation
- Evaluation workflow

---

## ⚠️ Limitations

- Speech recognition requires internet access
- Text-to-speech requires internet access
- Caption quality may vary depending on image complexity
- Currently supports English only
- Story generation output may vary between runs

---

## 🔮 Future Improvements

- Multi-language support
- Story genre selection
- Download generated stories as PDF/audio
- User authentication
- Better UI/UX enhancements
- Cloud deployment for public access

---

## 👩‍💻 Author

**Gunjan**  
B.Tech AIML Student | AI/ML Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.
