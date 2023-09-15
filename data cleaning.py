import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
 
def data_cleaning(text):
    # Lowercase all text
    text = str(text).lower()
   
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
   
    # Tokenize text
    words = word_tokenize(text)
   
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
   
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
 
    # Join words back into a single string
    cleaned_text = " ".join(words)
   
    return cleaned_text
 

# Load the sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")
 
df = pd.read_csv(r'filename', sep=',', engine='python', quotechar='"', error_bad_lines=False)
df['cleanse_data'] = df['Call Notes'].apply(data_cleaning)
df = df.dropna(how='any')
 

# Define a function to extract the sentiment label from the model output
def get_label(result):
    if result[0]["label"] == "POSITIVE":
        return "positive"
    elif result[0]["label"] == "NEGATIVE":
        return "negative"
    else:
        return "neutral"
   
# Apply the sentiment analysis model in parallel to each row of the dataset
df["label"] = df['cleanse_data'].apply(sentiment_analysis).apply(get_label)
 
df.to_csv(r'filename', index=False)