from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import numpy as np

# The mean pooling function for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sample sentences for testing
sample_sentences = [
    "The coffee was excellent but the service was slow.",
    "I love the ambience of this Starbucks location.",
    "The barista was extremely friendly and helpful.",
    "The barista took a lot of time to process my order, and still managed to get it wrong",
    "The coffee tasted terrible, and the ambience and dull",
    "I didn't like the overall experience",
    "I love how they handled my over-complexified order"
]

# Loading pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Generating embeddings
# 1. Tokenize the sentences
encoded_input = tokenizer(sample_sentences, padding=True, truncation=True, return_tensors='pt')

# 2. Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# 3. Apply mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# 4. Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Display results
print(f"Embedding shape: {sentence_embeddings.shape}")
print(f"First 5 dimensions of each sentence embedding:")
print(np.round(sentence_embeddings[:, :5].numpy(), 3))