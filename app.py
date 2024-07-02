import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import plotly.express as px
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import stopwords
import string

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def c_perplexity(text):
    """Calculate the perplexity of the given text using GPT-2."""
    if not text.strip():
        return float('inf')  # Return inf for empty input
    
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    if input_ids.size(1) == 0:  # Check for empty input after encoding
        return float('inf')

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()

def c_burstiness(text):
    """Calculate the burstiness of the given text."""
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0

    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    b_score = repeated_count / len(word_freq) if len(word_freq) > 0 else 0.0
    return b_score

def top_repword_count(text):
    """Generate a bar chart of the top 10 most repeated words."""
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    
    if not top_words:
        st.write("No significant words found.")
        return
    
    words, counts = zip(*top_words)
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title="Top 10 Most Repeated Words in the Text")
    st.plotly_chart(fig, user_container_width=True)

# Streamlit app configuration
st.set_page_config(layout="wide")

st.title("AI Content Detector")

text_area = st.text_area("Enter your text here!")

if text_area:
    if st.button("Analyse the content"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.info("Your input text")
            st.success(text_area)
            
        with col2:
            st.info("Your output score")
            perplexity = c_perplexity(text_area)
            burstiness = c_burstiness(text_area)
            
            st.success(f"Perplexity score: {perplexity}")
            st.success(f"Burstiness score: {burstiness}")
            
            if perplexity > 40000 or burstiness < 0.24:
                st.error("Result: The text is likely AI-generated.")
            else:
                st.success("Result: The text is not AI-generated.")
        
            st.warning("Disclaimer: AI plagiarism detector apps can assist in identifying potential instances of plagiarism.")
        
        with col3:
            st.info("Basic Review")
            top_repword_count(text_area)
