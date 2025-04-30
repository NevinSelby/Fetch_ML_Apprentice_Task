import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import json
import os
from model import StarbucksReviewAnalyzer, predict_review

st.set_page_config(
    page_title="Starbucks Reviews Analysis",
    page_icon="☕",
    layout="wide"
)

@st.cache_resource
def load_model_and_tokenizer(model_path, bert_model_name, num_attributes, num_sentiments):
    """Load model and tokenizer (cached to avoid reloading)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StarbucksReviewAnalyzer(
        bert_model_name=bert_model_name,
        num_attributes=num_attributes,
        num_sentiments=num_sentiments
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    return model, tokenizer, device

@st.cache_data
def load_mappings(attribute_map_path, sentiment_map_path):
    """Load label mappings (cached to avoid reloading)."""
    with open(attribute_map_path, 'r') as f:
        attribute_map = json.load(f)
        # Convert string keys back to integers
        idx_to_attribute = {int(k): v for k, v in attribute_map.items()}
        attribute_to_idx = {v: int(k) for k, v in attribute_map.items()}
    
    with open(sentiment_map_path, 'r') as f:
        sentiment_map = json.load(f)
        # Convert string keys back to integers
        idx_to_sentiment = {int(k): v for k, v in sentiment_map.items()}
        sentiment_to_idx = {v: int(k) for k, v in sentiment_map.items()}
    
    return {
        'attribute_map': {
            'idx_to_attribute': idx_to_attribute,
            'attribute_to_idx': attribute_to_idx
        },
        'sentiment_map': {
            'idx_to_sentiment': idx_to_sentiment,
            'sentiment_to_idx': sentiment_to_idx
        }
    }

@st.cache_data
def load_sample_reviews(csv_path, n=5):
    """Load a few sample reviews for demonstration."""
    df = pd.read_csv(csv_path)
    return df.sample(n=n, random_state=42)

def main():
    # Title and introduction
    st.title("☕ Starbucks's Reviews Analysis")
    st.markdown("""
    This application demonstrates a **Multi-Task Learning** approach to analyze Starbucks's customer reviews.
    The model simultaneously classifies:
    - **Attribute**: What aspect of Starbucks's the review discusses (service, food, etc.)  
    - **Sentiment**: The emotional tone of the review (positive, negative, neutral)
    """)
    
    # Load configurations
    model_path = "models/starbucks_mtl_best.pt"
    bert_model_name = "bert-base-uncased"
    data_path = "starbucks_reviews.csv"
    attribute_map_path = "results/attribute_map.json"
    sentiment_map_path = "results/sentiment_map.json"
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}.")
        return
    
    # Load mappings
    if os.path.exists(attribute_map_path) and os.path.exists(sentiment_map_path):
        mappings = load_mappings(attribute_map_path, sentiment_map_path)
    else:
        st.warning("Label mapping files not found. Using generic labels.")
        mappings = None
    
    # Determine number of classes from mappings
    if mappings:
        num_attributes = len(mappings['attribute_map']['idx_to_attribute'])
        num_sentiments = len(mappings['sentiment_map']['idx_to_sentiment'])
        
        # Display attribute and sentiment classes
        with st.expander("View Attribute and Sentiment Categories"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Attribute Categories")
                for idx, attr in mappings['attribute_map']['idx_to_attribute'].items():
                    st.write(f"{idx}: {attr}")
            
            with col2:
                st.subheader("Sentiment Categories")
                for idx, sent in mappings['sentiment_map']['idx_to_sentiment'].items():
                    st.write(f"{idx}: {sent}")
    else:
        # Default values if mappings not available
        num_attributes = 6  # Reasonable default
        num_sentiments = 3  # Typical sentiment classes
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        model_path, bert_model_name, num_attributes, num_sentiments
    )
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Live Prediction", "Sample Reviews", "Model Performance"])
    
    # Tab 1: Live Prediction
    with tab1:
        st.header("Analyze Your Own Review")
        review_text = st.text_area(
            "Enter a Starbucks review:",
            "The barista was very friendly and made a perfect caramel macchiato, but the store was too crowded and noisy.",
            height=100
        )
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                # Make prediction
                prediction = predict_review(
                    model, tokenizer, review_text, device, 
                    mappings['attribute_map'] if mappings else None,
                    mappings['sentiment_map'] if mappings else None
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Attribute Classification")
                    st.info(f"**Detected Attribute**: {prediction['attribute']['label']}")
                    
                    # Create attribute probability chart
                    attr_probs = prediction['attribute']['probabilities']
                    if mappings:
                        attr_labels = list(mappings['attribute_map']['idx_to_attribute'].values())
                    else:
                        attr_labels = [f"Class {i}" for i in range(len(attr_probs))]
                    
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(attr_probs))
                    ax.barh(y_pos, attr_probs, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(attr_labels)
                    ax.invert_yaxis()  # Labels read top-to-bottom
                    ax.set_title('Attribute Probabilities')
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Sentiment Analysis")
                    
                    # Set color based on sentiment
                    if prediction['sentiment']['label'] == 'Positive':
                        st.success(f"**Detected Sentiment**: {prediction['sentiment']['label']}")
                    elif prediction['sentiment']['label'] == 'Negative':
                        st.error(f"**Detected Sentiment**: {prediction['sentiment']['label']}")
                    else:
                        st.info(f"**Detected Sentiment**: {prediction['sentiment']['label']}")
                    
                    # Create sentiment probability chart
                    sent_probs = prediction['sentiment']['probabilities']
                    if mappings:
                        sent_labels = list(mappings['sentiment_map']['idx_to_sentiment'].values())
                    else:
                        sent_labels = [f"Class {i}" for i in range(len(sent_probs))]
                    
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(sent_probs))
                    ax.barh(y_pos, sent_probs, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(sent_labels)
                    ax.invert_yaxis()
                    ax.set_title('Sentiment Probabilities')
                    st.pyplot(fig)
    
    # Tab 2: Sample Reviews
    with tab2:
        st.header("Sample Reviews Analysis")
        
        try:
            sample_reviews = load_sample_reviews(data_path)
            
            for i, row in sample_reviews.iterrows():
                with st.expander(f"Review {i+1}: {row['review'][:100]}..."):
                    st.write(row['review'])
                    
                    col1, col2 = st.columns(2)
                    
                    # Ground truth (if available in dataset)
                    if 'attribute' in row and 'sentiment' in row:
                        with col1:
                            st.subheader("Ground Truth")
                            st.write(f"**Attribute**: {row['attribute']}")
                            st.write(f"**Sentiment**: {row['sentiment']}")
                    
                    # Model prediction
                    prediction = predict_review(
                        model, tokenizer, row['review'], device, 
                        mappings['attribute_map'] if mappings else None,
                        mappings['sentiment_map'] if mappings else None
                    )
                    
                    with col2:
                        st.subheader("Model Prediction")
                        st.write(f"**Attribute**: {prediction['attribute']['label']}")
                        st.write(f"**Sentiment**: {prediction['sentiment']['label']}")
        
        except Exception as e:
            st.error(f"Error loading sample reviews: {str(e)}")
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance")
        
        # Check if training history is available
        history_path = "results/training_history.csv"
        if os.path.exists(history_path):
            try:
                history = pd.read_csv(history_path)
                
                st.subheader("Training Curves")
                
                # Create performance visualizations
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Accuracy plots
                axes[0, 0].plot(history['train_attribute_acc'], label='Train')
                axes[0, 0].plot(history['val_attribute_acc'], label='Validation')
                axes[0, 0].set_title('Attribute Classification Accuracy')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].legend()
                
                axes[0, 1].plot(history['train_sentiment_acc'], label='Train')
                axes[0, 1].plot(history['val_sentiment_acc'], label='Validation')
                axes[0, 1].set_title('Sentiment Analysis Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                
                # F1 score plots
                axes[1, 0].plot(history['train_attribute_f1'], label='Train')
                axes[1, 0].plot(history['val_attribute_f1'], label='Validation')
                axes[1, 0].set_title('Attribute Classification F1 Score')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].legend()
                
                axes[1, 1].plot(history['train_sentiment_f1'], label='Train')
                axes[1, 1].plot(history['val_sentiment_f1'], label='Validation')
                axes[1, 1].set_title('Sentiment Analysis F1 Score')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('F1 Score')
                axes[1, 1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display final performance metrics
                last_epoch = len(history) - 1
                st.subheader("Final Model Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Attribute Accuracy", 
                        f"{history['val_attribute_acc'].iloc[last_epoch]:.4f}",
                        f"{history['val_attribute_acc'].iloc[last_epoch] - history['val_attribute_acc'].iloc[0]:.4f}"
                    )
                    st.metric(
                        "Attribute F1 Score", 
                        f"{history['val_attribute_f1'].iloc[last_epoch]:.4f}",
                        f"{history['val_attribute_f1'].iloc[last_epoch] - history['val_attribute_f1'].iloc[0]:.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Sentiment Accuracy", 
                        f"{history['val_sentiment_acc'].iloc[last_epoch]:.4f}",
                        f"{history['val_sentiment_acc'].iloc[last_epoch] - history['val_sentiment_acc'].iloc[0]:.4f}"
                    )
                    st.metric(
                        "Sentiment F1 Score", 
                        f"{history['val_sentiment_f1'].iloc[last_epoch]:.4f}",
                        f"{history['val_sentiment_f1'].iloc[last_epoch] - history['val_sentiment_f1'].iloc[0]:.4f}"
                    )
                
            except Exception as e:
                st.error(f"Error loading training history: {str(e)}")
        else:
            st.info("Training history not found. Train the model first to see performance metrics.")
        
        # Confusion matrices if available
        confusion_matrix_path = "results/confusion_matrices.png"
        if os.path.exists(confusion_matrix_path):
            st.subheader("Confusion Matrices")
            st.image(confusion_matrix_path)

    # Footer
    st.divider()
    st.caption("Created as part of a ML Apprentice take-home exercise. BERT-based multi-task learning model for Starbucks's reviews analysis.")

if __name__ == "__main__":
    main()
