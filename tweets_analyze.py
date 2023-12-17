# pip install pandas, numpy, gensim, nltk, scikit-learn, matplotlib   # first install if haven't

import pandas as pd
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# modify to local path if needed
csv_base_path = '../csv/IRAhandle_tweets_'
google_news_path = '~/Documents/02_Dev/bin/GoogleNews-vectors-negative300.bin'

# keywords for four types of biases
political_keywords   = ['democrat', 'republican', 'liberal', 'conservative', 'obama', 'trump']
gender_keywords      = ['woman', 'man', 'gender', 'equality', 'feminism', 'patriarchy']
religion_keywords    = ['christian', 'muslim', 'atheist', 'religion', 'faith', 'secular']
immigration_keywords = ['immigrant', 'border', 'refugee', 'asylum', 'deportation', 'visa']

all_keywords = political_keywords + gender_keywords + religion_keywords + immigration_keywords

def main():
    # -----------------------------
    # 1. Read CSV Files
    # -----------------------------
    tweets_files = []
    for i in range(1, 14):
        file_name = f"{csv_base_path}{i}.csv"
        tweets_files.append(file_name)

    columns_to_read = ['author', 'content', 'region', 'language', 'publish_date', 'account_type', 'account_category']

    # Read, concatenate, and filter for English language tweets
    all_tweets = pd.concat([pd.read_csv(file, usecols=columns_to_read) for file in tweets_files])
    english_tweets = all_tweets[all_tweets['language'] == 'English']
    tweets_content = english_tweets['content']

    # -----------------------------
    # 2. Clean & Tokenize
    # -----------------------------
    # Clean and Preliminary tokenization for phrase detection
    tokenized_tweets = [clean_text_and_tokenize(text) for text in tweets_content if pd.notnull(text)]

    # Detect phrases and apply
    phrased_tweets = detect_and_apply_phrases(tokenized_tweets)

    # Remove stopwords
    processed_tweets = [remove_stopwords(tweet) for tweet in phrased_tweets]
    print("PreProcessing tweets complete successfully!")

    # -----------------------------
    # 3. Train Word Embeddings
    # -----------------------------
    # Train the Word2Vec Model
    my_word2vec_model = Word2Vec(sentences=processed_tweets, vector_size=300, window=5, min_count=5, workers=4)
    print("Train Word Embeddings complete successfully!")

    # -----------------------------
    # 4. Compare with Pre-Trained 
    # -----------------------------
    # Load Google News Word2Vec model
    google_news_model = KeyedVectors.load_word2vec_format(google_news_path, binary=True)
    print("Pre-trained Google News model loaded successfully!")

    # Perform some basic comparisons for example word vector in both models
    word = 'democrat'
    if word in google_news_model.key_to_index and word in my_word2vec_model.wv.key_to_index:
        google_vector = google_news_model[word]
        my_vector = my_word2vec_model.wv[word]

        # Calculate cosine similarity
        similarity = cosine_similarity([google_vector], [my_vector])[0][0]
        print(f"Cosine similarity between '{word}' in both models: {similarity}")
    else:
        print(f"The word '{word}' is not in both models.")

    print("Load Pre-Trained word embeddings and compare successfully!")

    # -----------------------------
    # 5. Perform Spectral Clustering
    # -----------------------------
    # Extract embeddings
    my_model_embeddings = extract_embeddings(my_word2vec_model.wv, all_keywords)
    google_model_embeddings = extract_embeddings(google_news_model, all_keywords)

    # Combine embeddings from both models for clustering
    combined_embeddings = np.vstack((my_model_embeddings, google_model_embeddings))

    # Perform spectral clustering
    # Determine the number of clusters (k)
    num_clusters = 5
    clustering = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0)
    cluster_labels = clustering.fit_predict(combined_embeddings)

    print("Perform Spectral Clustering successfully!")

    # -----------------------------
    # 6. Visualize
    # -----------------------------
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=0)
    # Apply t-SNE to the combined embeddings
    reduced_embeddings = tsne.fit_transform(combined_embeddings)

    # Combine the four bias keywords into a bias_keywords dictionary
    bias_keywords = {
        'political': political_keywords,
        'gender': gender_keywords,
        'religion': religion_keywords,
        'immigration': immigration_keywords
    }

    # Calculate the offset for Google News embeddings
    offset = len(my_model_embeddings)

    # Plot for each bias type
    for bias_type, keywords in bias_keywords.items():
        plot_bias_type(reduced_embeddings, cluster_labels, bias_type, keywords, offset, f't-SNE visualization of {bias_type} bias')



# Clean texts and preliminary tokenization
def clean_text_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\S+', '', text)   # Remove Tweeter Handles (content between @ and the next space)
    text = re.sub(r'\#\S+', '', text)   # Remove Hashtags (content between # and the next space)
    text = re.sub(r'\d+', '', text)     # Remove numbers

    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642" 
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'\s+',' ',text)  # Replace multiple spaces with a single space
    text = text.strip()             # Remove leading and trailing spaces

    return word_tokenize(text)

# Phrase Detection and Application
def detect_and_apply_phrases(tokenized_tweets):
    # Train the phrase model
    phrases = Phrases(tokenized_tweets, min_count=3, threshold=10)
    bigram_model = Phraser(phrases)
    
    # Apply the phrase model
    phrased_tweets = [bigram_model[tweet_tokens] for tweet_tokens in tokenized_tweets]
    return phrased_tweets

# Remove stopwords
def remove_stopwords(tweet_tokens):
    return [word for word in tweet_tokens if word not in stop_words]

# Extract keyword embeddings from models
def extract_embeddings(model, keywords):
    embeddings = []
    for word in keywords:
        if word in model.key_to_index:
            embeddings.append(model[word])
    return embeddings

# Plotting figure for a specific bias type
def plot_bias_type(reduced_embeddings, cluster_labels, bias_type, keywords, offset, title):

    # Markers for each model
    model_markers = {
        'russian_troll_model': 'o',  # Circle marker for Russian Troll Model
        'google_news_model': 'x',    # Cross marker for Google News Model
    }

    # Colormap for clusters
    cluster_colormap = plt.cm.get_cmap('viridis', len(np.unique(cluster_labels)))

    plt.figure(figsize=(10, 8))
    
    # Plot embeddings from Russian Troll Model
    for word in keywords:
        if word in all_keywords:
            index = all_keywords.index(word)
            # Use cluster label to determine color
            color = cluster_colormap(cluster_labels[index])
            plt.scatter(reduced_embeddings[index, 0], reduced_embeddings[index, 1],
                        color=color, marker=model_markers['russian_troll_model'],
                        label='Russian Troll Model' if word == keywords[0] else "")
            plt.text(reduced_embeddings[index, 0], reduced_embeddings[index, 1], word)
    
    # Plot embeddings from Google News model
    for word in keywords:
        if word in all_keywords:
            index = all_keywords.index(word) + offset
            # Use cluster label to determine color
            color = cluster_colormap(cluster_labels[index])
            plt.scatter(reduced_embeddings[index, 0], reduced_embeddings[index, 1],
                        color=color, marker=model_markers['google_news_model'], 
                        label='Google News Model' if word == keywords[0] else "")
            plt.text(reduced_embeddings[index, 0], reduced_embeddings[index, 1], word)
    
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    plt.legend(handles=[
        plt.Line2D([0], [0], marker=model_markers['russian_troll_model'], label='Russian Troll Model',
                   markersize=7, linestyle='None'),
        plt.Line2D([0], [0], marker=model_markers['google_news_model'], label='Google News Model',
                   markersize=6, linestyle='None')
    ])

    plt.show()


if __name__ == '__main__':
    main()

