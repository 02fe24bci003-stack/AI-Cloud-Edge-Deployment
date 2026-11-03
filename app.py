
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time

app = FastAPI(title="AI Cloud and Edge Application")

# Sample training data
texts = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "Excellent service",
    "I hate this product",
    "This is very bad",
    "I am disappointed",
    "Terrible experience",
    "The product is okay",
    "It is average",
    "Nothing special"
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Negative",
    "Neutral",
    "Neutral",
    "Neutral"
]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train AI model
model = MultinomialNB()
model.fit(X, labels)


# Input format
class UserInput(BaseModel):
    text: str


# Home API
@app.get("/")
def home():
    return {
        "message": "AI Cloud and Edge Application is running"
    }


# Sentiment prediction API
@app.post("/predict")
def predict(data: UserInput):

    start_time = time.time()

    input_data = vectorizer.transform([data.text])
    prediction = model.predict(input_data)[0]

    end_time = time.time()

    response_time = (end_time - start_time) * 1000

    return {
        "input_text": data.text,
        "predicted_sentiment": prediction,
        "response_time_ms": round(response_time, 3)
    }
