import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
from io import StringIO
import os

# Load SentenceTransformer model (replacing spaCy)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast, and effective model

# Mock "previously used topics" dataset (could be replaced with a file or database in practice)
PREVIOUS_TOPICS = [
    "I need a 5000 rs personal loan",
    "How to apply for a home loan",
    "Can I get a 20000 rs business loan",
    "Best personal loan rates",
]

# Function to extract topics from different input types
def extract_topics(uploaded_file=None, text_input=None):
    topics = []
    
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.txt':
            content = uploaded_file.read().decode("utf-8")
            topics = [line.strip() for line in content.split('\n') if line.strip()]
        
        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            if 'topic' in df.columns:
                topics = df['topic'].dropna().tolist()
            else:
                topics = df.iloc[:, 0].dropna().tolist()
        
        elif file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
            if 'topic' in df.columns:
                topics = df['topic'].dropna().tolist()
            else:
                topics = df.iloc[:, 0].dropna().tolist()
    
    if text_input:
        topics.extend([line.strip() for line in text_input.split('\n') if line.strip()])
    
    # Remove duplicates while preserving order
    topics = list(dict.fromkeys(topics))
    return topics

# Function to compute similarity and assign scores
def compute_similarity_and_score(topics, previous_topics):
    all_topics = topics + previous_topics
    embeddings = model.encode(all_topics, convert_to_numpy=True)
    similarity_matrix = cosine_similarity(embeddings)
    
    # Results for new topics comparison
    results = []
    topic_scores = {topic: 0 for topic in topics}  # Initialize scores
    
    # Compare new topics among themselves
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            similarity_score = similarity_matrix[i][j] * 100
            if similarity_score >= 70:
                results.append({
                    'Topic 1': topics[i],
                    'Topic 2': topics[j],
                    'Similarity (%)': round(similarity_score, 2),
                    'Warning': 'Potential Cannibalization'
                })
            topic_scores[topics[i]] = max(topic_scores[topics[i]], similarity_score)
            topic_scores[topics[j]] = max(topic_scores[topics[j]], similarity_score)
    
    # Compare new topics with previous topics
    for i, topic in enumerate(topics):
        for j in range(len(topics), len(all_topics)):
            similarity_score = similarity_matrix[i][j] * 100
            if similarity_score >= 70:
                results.append({
                    'Topic 1': topic,
                    'Topic 2': previous_topics[j - len(topics)],
                    'Similarity (%)': round(similarity_score, 2),
                    'Warning': 'Matches Previous Topic'
                })
            topic_scores[topic] = max(topic_scores[topic], similarity_score)
    
    return pd.DataFrame(results), topic_scores

# Function to create a network graph of similar topics
def create_network_graph(topics, similarity_df):
    G = nx.Graph()
    for topic in topics:
        G.add_node(topic)
    for _, row in similarity_df.iterrows():
        G.add_edge(row['Topic 1'], row['Topic 2'], weight=row['Similarity (%)'])
    
    pos = nx.spring_layout(G)
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[2]['weight']:.2f}%")
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='text', text=edge_text, mode='lines')
    
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text, textposition="top center",
                            marker=dict(showscale=False, color='skyblue', size=10, line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='Network Graph of Similar Topics', titlefont_size=16, showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# Streamlit app
st.title("Topic Cannibalization Checker with Scoring")
st.write("Upload a file (TXT, CSV, or Excel) or paste topics to check for similarity, previous usage, and get scores.")

uploaded_file = st.file_uploader("Upload a file", type=['txt', 'csv', 'xlsx'])
text_input = st.text_area("Or paste topics here (one per line, e.g., 'I want 10000 rs personal loan')", height=200)

if st.button("Check for Cannibalization"):
    if not uploaded_file and not text_input:
        st.error("Please upload a file or enter topics.")
    else:
        with st.spinner("Processing topics..."):
            topics = extract_topics(uploaded_file, text_input)
            
            if len(topics) == 0:
                st.error("No valid topics found in the input.")
            else:
                st.write(f"Found {len(topics)} unique topics.")
                
                # Compute similarity and scores
                similarity_df, topic_scores = compute_similarity_and_score(topics, PREVIOUS_TOPICS)
                
                # Display scores with animation-like effect
                st.subheader("Topic Scores")
                for topic, score in topic_scores.items():
                    if score >= 70:
                        st.markdown(f"**{topic}**: {score:.2f}% similarity ⚠️ *[Potential Issue]*")
                        st.markdown("<style>@keyframes blink {0% {color: red;} 50% {color: yellow;} 100% {color: red;}} "
                                    ".blink {animation: blink 1s infinite;}</style>", unsafe_allow_html=True)
                        st.markdown(f"<span class='blink'>High Similarity Detected!</span>", unsafe_allow_html=True)
                    elif score >= 60:
                        st.markdown(f"**{topic}**: {score:.2f}% similarity ⚠️ *[接近70%]*")
                        st.markdown("<style>@keyframes pulse {0% {color: orange;} 50% {color: white;} 100% {color: orange;}} "
                                    ".pulse {animation: pulse 1.5s infinite;}</style>", unsafe_allow_html=True)
                        st.markdown(f"<span class='pulse'>Approaching Threshold</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{topic}**: {score:.2f}% similarity ✅")

                if similarity_df.empty:
                    st.success("No topics with similarity ≥70% found. No cannibalization detected!")
                else:
                    st.warning("Potential cannibalization or overlap with previous topics detected!")
                    
                    st.subheader("Similar Topics")
                    st.dataframe(similarity_df.style.highlight_max(subset=['Similarity (%)'], color='yellow'))
                    
                    csv = similarity_df.to_csv(index=False)
                    st.download_button(label="Download Results as CSV", data=csv, file_name="similar_topics.csv", mime="text/csv")
                    
                    st.subheader("Topic Similarity Network")
                    fig = create_network_graph(topics, similarity_df)
                    st.plotly_chart(fig)