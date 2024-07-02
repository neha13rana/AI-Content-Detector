# **AIcontentdetector** (https://aicontentdetector.streamlit.app/)

In this project, I have developed an AI content detector using perplexity and burstiness concepts in NLP.

**Perplexity:** Perplexity measures how well a probability model predicts a sample, particularly in Natural Language Processing (NLP). A language model, which generates and evaluates sentences, should assign higher probabilities to well-written texts. Perplexity thus captures a model's uncertainty in predicting text. For example, given a trained language model that predicts words from a limited set, the probability of the sentence "a red fox." is calculated by multiplying the probabilities of each word conditional on its predecessors: P("a red fox.") = P("a") * P("red" | "a") * P("fox" | "a red") * P("." | "a red fox").

**Burstiness:** In a unigram model, the distribution of a word is evenly spread out across events (words) and could be represented as a repeated Bernoulli trial with probability P(w). This model works for most functional words, but content words have different distributions, for which a bigram model is used.
Here I'm using the GPT-2 transformer. User can check if the text contains AI content and replace it with the provided review when analyzing the text to reduce the chances of AI content.

 
**Used technology :**
1) Streanlit
2) Streamlit.io for deployment
3) Machine Learning
4) NLP (standard measures of perplexity and burstiness and preprocessing of the model)
5) Data visualization
6) Pytorch
7) Matplotlib

<hr>

**Other model training on the AI_Human dataset involves the following steps:**
1) NLP for data preprocessing, 
2) EDA, 
3) Created a pipeline which includes CountVectorizer, TfidfTransformer, and MultinomialNB model.
The accuracy achieved is 95%.

![image](https://github.com/neha13rana/AIcontentdetector/assets/121093178/218fd212-0e51-49bd-866e-dee684ba5448)


<hr>

**1) paste the text from the public pdf.**

Result : Ai content not detected in the text.

![WhatsApp Image 2024-06-29 at 13 56 53_24ff75e7](https://github.com/neha13rana/AIcontentdetector/assets/121093178/38ae1fe9-934c-4e06-af13-43a5fad6d9b4)

**2) Text is generated by chatgpt :**

Result : Ai content detected in the text. review shows that in top 10 word you have a ai content change that content to get the less content of AI.
![image](https://github.com/neha13rana/AIcontentdetector/assets/121093178/30e8694c-e73c-4c78-b903-0d308a96a40a)

 **Quilbot result :**

![WhatsApp Image 2024-06-29 at 13 57 29_ce29cda6](https://github.com/neha13rana/AIcontentdetector/assets/121093178/2617892e-7b97-4c26-9d36-327f2d642402)

**3) download the review :**

![image](https://github.com/neha13rana/AIcontentdetector/assets/121093178/2e0da58d-9f5e-41e6-ae75-e98d4256b04b)


<hr> 

**Steps :**

1) Download the files from my gtihub account https://github.com/neha13rana/AIcontentdetector or just clone the website by using git command git clone https://github.com/neha13rana/AIcontentdetector.git if you have a set up of git in your device.
  
2) set the virtual environment by writing :  1. python -m venv venv   2.  venv\Scripts\activate

3) than install requirements.txt to install the depencies 3. pip install -r requirements.txt .

4) after installing run your app by using the command 4. streamlit run app.py

5) open the local/network URL on your browser.

6) Enter your text and check if your content is ai free or not and want to decrease the amount of ai content than change the words by using the review provided by the site review.

**Resources :**
Kaggle 

https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584

https://nlp.fi.muni.cz/raslan/2011/paper17.pdf
