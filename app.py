import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải các tài nguyên NLTK cần thiết
nltk.download('stopwords')
nltk.download('wordnet')
emojis = pd.read_csv('./dataset/emojis.txt', sep=',', header=None)
emojis_dict = {i: j for i, j in zip(emojis[0], emojis[1])}
pattern = '|'.join(sorted(re.escape(k) for k in emojis_dict))

def replace_emojis(text):
    if emojis_dict:
        text = re.sub(pattern, lambda m: emojis_dict.get(m.group(0)), text, flags=re.IGNORECASE)
    return text

def remove_punct(text):
    text = replace_emojis(text)
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = text.lower()
    text = re.split(r'\W+', text)
    return text

def remove_stopwords(text):
    stopword = stopwords.words('english')
    stopword.extend(['yr', 'year', 'woman', 'man', 'girl', 'boy', 'one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 
                     'week', 'treatment', 'associated', 'patients', 'may', 'day', 'case', 'old', 'u', 'n', 'didnt', 
                     'ive', 'ate', 'feel', 'keep', 'brother', 'dad', 'basic', 'im'])
    text = [word for word in text if word not in stopword]
    return text

def lemmatizer(text):
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]
    return text

def clean_text(text):
    text = remove_punct(text)
    text = tokenization(text)
    text = remove_stopwords(text)
    text = lemmatizer(text)
    return text

# Load model và vectorizer
model = joblib.load('./models/best_emotion_model.pkl')
count_vectorizer = joblib.load('./models/count_vectorizer.pkl')
tfidf_transformer = joblib.load('./models/tfidf_transformer.pkl')


# Cấu hình emoji
emotions_emoji_dict = {
    "anger": "😠",  
    "fear": "😱", 
    "joy": "😂", 
    "love": "🥰",
    "sadness": "😔",
    "surprise": "😮"
}

# Thiết lập giao diện
st.set_page_config(
    page_title="Phân tích Cảm xúc",
    page_icon="🤖",
    layout="centered"
)

st.title("Emotion Detection In Text")

# Căn giữa phần text input
st.markdown("Enter your text")
user_input = st.text_area("Input Text:", placeholder="Type your text here...", height=150, label_visibility="collapsed")

# Căn giữa nút Analyze
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    analyze_clicked = st.button("Analyze Text 🔍", type="primary", use_container_width=True)

if analyze_clicked:
    if user_input:
        # Bước 1: CountVectorizer
        text_count = count_vectorizer.transform([user_input])
        # Bước 2: TF-IDF transformation
        text_tfidf = tfidf_transformer.transform(text_count)
        # Bước 3: Dự đoán
        probabilities = model.predict_proba(text_tfidf)[0]
        predicted_index = np.argmax(probabilities)
        prediction = model.classes_[predicted_index] 
        max_prob = round(probabilities[predicted_index] * 100, 2)
        
        # Hiển thị kết quả trong giao diện
        col_result, col_details = st.columns([1, 2])
        
        with col_result:
            st.subheader("Analysis Result")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.metric(label="Predicted Emotion", 
                      value=f"{prediction.capitalize()} {emoji_icon}",
                      delta=f"Confidence: {max_prob}%")
        
        with col_details:
            st.subheader("Detailed Probabilities")
            for emotion, prob in zip(model.classes_, probabilities):
                progress = int(round(prob, 2) * 100)
                emoji = emotions_emoji_dict.get(emotion, "")
                percentage = f"{round(prob*100, 2):.2f}%"
                
                cols = st.columns([2, 4, 1])
                with cols[0]:
                    st.markdown(f"**{emotion.capitalize()}** {emoji}", help=f"Probability: {percentage}")
                with cols[1]:
                    st.progress(progress)
                with cols[2]:
                    st.markdown(f"`{percentage}`")
    else:
        st.warning("Please input text before analyzing!")
