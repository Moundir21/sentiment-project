# Algerian Dialect Sentiment Analysis API

AI project for **Sentiment Analysis of Algerian Arabic comments** using Deep Learning and Transformer models.

The system analyzes user comments and classifies them into:

* Positive 😀
* Neutral 😐
* Negative 😡

The model is designed to handle **Arabic text, Algerian dialect, emojis, and informal writing used on social media**.

---

# Project Goal

The goal of this project is to build an **Artificial Intelligence API** that can analyze the sentiment of comments written in Arabic or Algerian dialect.

This type of system can be used for:

* Customer feedback analysis
* Social media monitoring
* Product review analysis
* Public opinion analysis

---

# Technologies Used

This project was built using the following technologies:

### Programming Language

* Python

### AI / Machine Learning

* PyTorch
* Transformers (HuggingFace)
* AraBERT model

### NLP Libraries

* arabert
* camel-tools
* emoji

### Backend API

* FastAPI

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

---

# AI Model

The project uses the **AraBERT Transformer model** for Arabic Natural Language Processing.

Model used:

`aubmindlab/bert-base-arabertv02`

Why AraBERT?

* trained on Arabic text
* understands Arabic grammar
* better performance for Arabic sentiment analysis
* supports dialect better than multilingual BERT

---

# Features

The system performs several processing steps before prediction.

### Text Cleaning

The system removes:

* URLs
* hashtags
* mentions
* special characters
* repeated letters

Example:

Before

الخدمة رااااائعة 😍😍

After

الخدمة رائعة smiling_face_with_heart_eyes

---

### Emoji Processing

Emojis are converted into words so the model understands the sentiment.

Example

😍 → smiling_face_with_heart_eyes

😡 → angry_face

---

### Arabic Normalization

Arabic letters are normalized:

Example

إ أ آ → ا

ة → ه

ى → ي

This improves the model learning process.

---

# Project Structure

```
sentiment-project/

backend/

│
├── api.py
│ FastAPI server for predictions
│
├── preprocess.py
│ Text cleaning and normalization
│
├── transformer_model.py
│ Loads AraBERT model and tokenizer
│
├── train_transformer.py
│ Script used to train the AI model
│
├── evaluate.py
│ Evaluates the model accuracy
│
├── data/
│ Dataset used for training
│
├── arabert_sentiment_model/
│ Trained model files
│
├── requirements.txt
│ Python dependencies
│
└── README.md
│ Project documentation
```

---

# Installation

Clone the project:

```
git clone https://github.com/yourusername/sentiment-project.git
```

Enter the project folder:

```
cd sentiment-project/backend
```

Install required libraries:

```
pip install -r requirements.txt
```

or manually:

```
pip install pandas numpy scikit-learn torch transformers fastapi uvicorn arabert camel-tools emoji matplotlib seaborn joblib
```

---

# Training the Model

To train the AI model run:

```
python train_transformer.py
```

The training process will:

1. Load dataset
2. Clean text
3. tokenize sentences
4. train the AraBERT model
5. save the trained model

The model will be saved in:

```
arabert_sentiment_model/
```

---

# Running the API

Start the FastAPI server:

```
uvicorn api:app --reload
```

The server will run on:

```
http://localhost:8000
```

API documentation:

```
http://localhost:8000/docs
```

---

# Example API Request

Using curl:

```
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-H "x-api-key: demo-key-1234" \
-d '{"text":"الخدمة رائعة"}'
```

Example response:

```
{
 "sentiment": "positive",
 "confidence": 0.93
}
```

---

# Model Evaluation

The model can be evaluated using:

```
python evaluate.py
```

The evaluation script calculates:

* Accuracy
* Precision
* Recall
* F1-score

This helps measure the model performance.

---

# Updating the Project on GitHub

Navigate to the project directory:

```
cd ~/myprojects/python/sentiment-project
```

Check changes:

```
git status
```

Add files:

```
git add .
```

Commit changes:

```
git commit -m "Update sentiment model and backend API"
```

Push to GitHub:

```
git push origin main
```

or

```
git push origin master
```

---

# Requirements File

Example `requirements.txt`

```
pandas
numpy
scikit-learn
torch
transformers
fastapi
uvicorn
arabert
camel-tools
emoji
matplotlib
seaborn
joblib
```

---

# Author

Mandir Badaj
University of Skikda
Computer Science Student

---

# License

This project is for **educational and research purposes**.
