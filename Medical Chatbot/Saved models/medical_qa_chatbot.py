import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy medical entity model
nlp = spacy.load("en_core_web_sm")

# Fixed dataset path
DATASET_PATH = "C:\\Users\\Aditya\\Desktop\\mustu bhai\\Medical Chatbot\\MedQuAD-master"
CACHE_FILE = "cached_medquad_data.csv"  # Store processed data

@st.cache_data
def load_medquad_cached():
    """Load data from cache if available, otherwise process XML files."""
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    else:
        df = load_medquad_from_xml(DATASET_PATH)
        df.to_csv(CACHE_FILE, index=False)  # Save processed data
        return df

def load_medquad_from_xml(root_folder):
    """Parse multiple folders containing XML files and extract questions & answers."""
    data = []
    
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            for file in os.listdir(folder_path):
                if file.endswith(".xml"):
                    file_path = os.path.join(folder_path, file)
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    
                    for qa in root.findall(".//QAPair"):
                        question_elem = qa.find("Question")
                        answer_elem = qa.find("Answer")

                        question = question_elem.text.strip() if question_elem is not None and question_elem.text else None
                        answer = answer_elem.text.strip() if answer_elem is not None and answer_elem.text else None

                        if question and answer:  # Ensure both are valid
                            data.append({"question": question, "answer": answer})
    
    return pd.DataFrame(data)

def preprocess_text(text):
    """Basic text preprocessing."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

@st.cache_data
def train_retrieval_model(df):
    """Train TF-IDF retrieval model."""
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(df['question'])
    return vectorizer, tfidf_matrix

def retrieve_answer(user_question, df, vectorizer, tfidf_matrix):
    """Retrieve the most relevant answer based on user question."""
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    best_match_idx = similarities.argmax()
    return df.iloc[best_match_idx]['answer']

def recognize_entities(text):
    """Perform simple entity recognition (symptoms, diseases, treatments)."""
    doc = nlp(text)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# Streamlit UI Enhancements
st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="💬", layout="centered")
st.markdown("""
    <style>
        body {background-color: #121212; color: white;}
        .stTextInput {border-radius: 8px;}
        .stTextInput input {padding: 10px; font-size: 16px;}
        .stButton>button {border-radius: 8px; font-size: 16px; background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 Medical Q&A Chatbot")

st.success("✅ Loading MedQuAD dataset from cache...")

df = load_medquad_cached()

if not df.empty:
    vectorizer, tfidf_matrix = train_retrieval_model(df)
    
    user_input = st.text_input("🩺 Ask a medical question:", key="user_question")
    
    while user_input:
        answer = retrieve_answer(user_input, df, vectorizer, tfidf_matrix)
        entities = recognize_entities(user_input)

        st.markdown("""---""")
        st.subheader("📝 Your Question:")
        st.info(user_input)
        
        st.subheader("💡 Answer:")
        st.success(answer)

        if entities:
            st.subheader("🔍 Identified Medical Entities:")
            for label, texts in entities.items():
                st.warning(f"**{label}:** {', '.join(texts)}")

        # Follow-up question prompt
        st.subheader("🤔 Follow-up:")
        st.write("Do you have any other questions? Feel free to ask!")
        user_input = st.text_input("💬 Ask another question:", key=f"followup_question_{len(user_input)}")
else:
    st.error("❌ No valid question-answer pairs found in the dataset.")
