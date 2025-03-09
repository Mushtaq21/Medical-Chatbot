import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy medical entity model (ensure you have installed an appropriate model)
nlp = spacy.load("en_core_web_sm")  # Replace with a medical-specific model if available

def load_medquad_from_xml(folder_path):
    """Parse multiple XML files and extract questions & answers."""
    data = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".xml"):
            file_path = os.path.join(folder_path, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for qa in root.findall(".//QAPair"):
                question = qa.find("Question").text.strip() if qa.find("Question") is not None else None
                answer = qa.find("Answer").text.strip() if qa.find("Answer") is not None else None
                
                if question and answer:
                    data.append({"question": question, "answer": answer})
    
    return pd.DataFrame(data)

def preprocess_text(text):
    """Basic text preprocessing."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

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

# Streamlit UI
st.title("Medical Q&A Chatbot")

folder_path = st.text_input("Enter the path to MedQuAD XML folder:")

df, vectorizer, tfidf_matrix = None, None, None

if folder_path:
    if os.path.isdir(folder_path):
        st.success("Processing MedQuAD dataset...")
        df = load_medquad_from_xml(folder_path)
        vectorizer, tfidf_matrix = train_retrieval_model(df)
        
        user_input = st.text_input("Ask a medical question:")
        if user_input and df is not None:
            answer = retrieve_answer(user_input, df, vectorizer, tfidf_matrix)
            entities = recognize_entities(user_input)

            st.subheader("Answer:")
            st.write(answer)

            if entities:
                st.subheader("Identified Medical Entities:")
                for label, texts in entities.items():
                    st.write(f"**{label}:** {', '.join(texts)}")
    else:
        st.error("Invalid folder path. Please enter a valid MedQuAD dataset directory.")
