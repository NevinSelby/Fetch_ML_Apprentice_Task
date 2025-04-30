import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
import re

class McDonaldsReviewsDataset(Dataset):
    """Dataset class for McDonald's reviews with both attribute and sentiment labels."""
    def __init__(self, reviews, attribute_labels, sentiment_labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.attribute_labels = attribute_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        """Process a single review into model-ready format with both task labels."""
        review = str(self.reviews[idx])
        
        # Tokenize the review text
        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return a dictionary with all required inputs and labels
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attribute_label': torch.tensor(self.attribute_labels[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }

def preprocess_text(text):
    """Clean review text by removing special characters and normalizing whitespace."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_ratings_to_sentiment(rating):
    """Convert numerical ratings to sentiment categories."""
    try:
        rating = float(rating)  # Convert to float in case of decimals
    except ValueError:
        return 'neutral'
    
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

def extract_attribute_from_review(review, category=None):
    """Extracts likely attribute from review text using keyword matching.
    
    This is a simplified approach. In a production environment, you might want
    to use a more sophisticated NLP approach or manual labeling.
    """
    review = review.lower()
    
    # Define attribute keywords
    attribute_keywords = {
        'food': ['food', 'burger', 'fries', 'nugget', 'mcnugget', 'sandwich', 'breakfast', 'meal', 'chicken', 'taste', 'delicious'],
        'service': ['service', 'staff', 'cashier', 'employee', 'server', 'worker', 'wait', 'time', 'fast', 'slow'],
        'cleanliness': ['clean', 'dirty', 'mess', 'hygiene', 'sanitize', 'tidy', 'swept', 'mopped', 'bathroom', 'toilet'],
        'price': ['price', 'expensive', 'cheap', 'afford', 'cost', 'money', 'dollar', 'value'],
        'atmosphere': ['atmosphere', 'environment', 'ambiance', 'seat', 'table', 'chair', 'noise', 'loud', 'quiet'],
        'location': ['location', 'parking', 'drive-thru', 'drive through', 'access', 'area', 'neighborhood']
    }
    
    # Count keyword matches for each attribute
    attribute_counts = {attr: 0 for attr in attribute_keywords}
    
    for attr, keywords in attribute_keywords.items():
        for keyword in keywords:
            if keyword in review:
                attribute_counts[attr] += 1
    
    # If category is provided, give it a slight boost
    if category and category.lower() in attribute_counts:
        attribute_counts[category.lower()] += 0.5
    
    # Return the attribute with the most keyword matches
    # If no matches found, default to 'food' as most common topic
    max_count = max(attribute_counts.values())
    if max_count > 0:
        for attr, count in attribute_counts.items():
            if count == max_count:
                return attr
    
    return 'food'  # Default attribute if no keywords matched

def load_and_prepare_data(csv_path, tokenizer_name='bert-base-uncased', test_size=0.2, val_size=0.1, 
                          batch_size=32, random_state=42):
    """Load, preprocess, and prepare McDonald's reviews data for multi-task learning."""
    # Load data
    df = pd.read_csv(csv_path, encoding='latin1')
    
    # Basic preprocessing
    df['review_clean'] = df['review'].apply(preprocess_text)
    
    # Convert ratings to sentiment classes
    df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
    
    # Extract attributes from review content
    df['attribute'] = df.apply(lambda row: extract_attribute_from_review(row['review_clean'], row.get('category')), axis=1)
    
    # Create attribute and sentiment label mappings
    attribute_categories = df['attribute'].unique()
    attribute_to_idx = {attr: idx for idx, attr in enumerate(attribute_categories)}
    idx_to_attribute = {idx: attr for attr, idx in attribute_to_idx.items()}
    
    sentiment_categories = df['sentiment'].unique()
    sentiment_to_idx = {sent: idx for idx, sent in enumerate(sentiment_categories)}
    idx_to_sentiment = {idx: sent for sent, idx in sentiment_to_idx.items()}
    
    # Convert labels to indices
    df['attribute_idx'] = df['attribute'].map(attribute_to_idx)
    df['sentiment_idx'] = df['sentiment'].map(sentiment_to_idx)
    
    # Split data: first into train+val and test, then train and val
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[['attribute', 'sentiment']]
    )
    
    # Calculate validation size relative to train+val
    relative_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, 
        random_state=random_state, stratify=train_val_df[['attribute', 'sentiment']]
    )
    
    print(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = McDonaldsReviewsDataset(
        train_df['review_clean'].values,
        train_df['attribute_idx'].values,
        train_df['sentiment_idx'].values,
        tokenizer
    )
    
    val_dataset = McDonaldsReviewsDataset(
        val_df['review_clean'].values,
        val_df['attribute_idx'].values,
        val_df['sentiment_idx'].values,
        tokenizer
    )
    
    test_dataset = McDonaldsReviewsDataset(
        test_df['review_clean'].values,
        test_df['attribute_idx'].values,
        test_df['sentiment_idx'].values,
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Return everything needed for training and evaluation
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'attribute_map': {'idx_to_attribute': idx_to_attribute, 'attribute_to_idx': attribute_to_idx},
        'sentiment_map': {'idx_to_sentiment': idx_to_sentiment, 'sentiment_to_idx': sentiment_to_idx},
        'num_attributes': len(attribute_categories),
        'num_sentiments': len(sentiment_categories),
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        },
        'dataframes': {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    }
