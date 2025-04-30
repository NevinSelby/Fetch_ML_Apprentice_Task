# Starbucks Reviews Multi-Task Learning Analysis

## Project Overview

This project implements a comprehensive multi-task learning system for analyzing Starbucks customer reviews. Using BERT as the backbone architecture, the model simultaneously performs two critical NLP tasks: attribute classification (identifying which aspect of Starbucks the review discusses - ambiance, food, location, service, price, or general) and sentiment analysis (determining if the sentiment is positive, neutral, or negative). This approach leverages shared representations while allowing for task-specific predictions, creating a more efficient and powerful analysis tool.

## Tasks Completed

I've successfully completed all the required tasks from the ML Apprentice Take-Home Exercise:

1. **Sentence Transformer Implementation**: Created a BERT-based sentence encoder that generates fixed-length embeddings
2. **Multi-Task Learning Expansion**: Extended the model to handle both attribute classification and sentiment analysis simultaneously
3. **Training Considerations**: Implemented strategic freezing/unfreezing approaches for effective transfer learning
4. **Training Loop Implementation**: Developed a complete multi-task training pipeline with dynamic loss weighting

Plus, I've added a Streamlit web interface for easy visualization and interaction with the model!

## Implementation Approach \& Thought Process

### Task 1: Sentence Transformer Implementation

I decided to use BERT for my sentence transformer implementation because of its robust contextual understanding of text. While I considered lighter alternatives like all-MiniLM-L6-v2, I ultimately chose BERT because:

1. I'm more familiar with BERT's architecture and behavior
2. Its 768-dimensional embeddings provide rich semantic representations for downstream tasks
3. It's widely used and recognized in industry, making my implementation more accessible

For the pooling strategy, I implemented both CLS token and mean pooling approaches but defaulted to mean pooling as it typically performs better for sentence-level tasks without further fine-tuning. The pooling layer is crucial because raw BERT outputs token-level embeddings that need to be aggregated into a single sentence representation.

One interesting challenge was handling padding tokens in the mean pooling calculation - I needed to explicitly mask these tokens out to prevent them from skewing the sentence embeddings.

### Task 2: Multi-Task Learning Expansion

Extending to multi-task learning was both challenging and exciting! I created a model with:

- A shared BERT backbone that learns common language features
- Two task-specific classification heads:
    - Attribute classifier: Categorizes reviews into Starbucks-specific aspects (ambiance, food, location, service, price, general)
    - Sentiment analyzer: Determines if the sentiment is positive, neutral, or negative

I made the architectural choice to use similar structures for both task heads (linear → ReLU → dropout → linear) because both tasks involve sentence-level classification. Each head has enough parameters to learn task-specific patterns while sharing the computationally expensive BERT backbone.

The most interesting part was designing the interface between the shared and task-specific components. I chose to use BERT's pooler_output as this interface since it's already designed to capture sentence-level semantics.

### Task 3: Training Considerations

Transfer learning is critical for BERT-based models, so I carefully considered different freezing strategies:

1. **Freezing the entire network**: This would be too restrictive, preventing the model from adapting to the Starbucks domain language
2. **Freezing only the BERT backbone**: This approach makes sense for limited data scenarios, letting the task heads adapt while preserving language fundamentals
3. **Freezing one task head**: This would be useful if one task was pre-trained but would likely cause the backbone to overfit to the unfrozen task

I implemented a gradual unfreezing approach where:

- Initially, the BERT backbone is completely frozen
- After the first epoch, the last few layers are selectively unfrozen
- A lower learning rate is used for fine-tuning to prevent catastrophic forgetting

This approach strikes a balance between adaptation and preservation of pre-trained knowledge, which is crucial for the Starbucks reviews domain where specific coffee, food, and café terminology might not be well-represented in BERT's original training data.

### Task 4: Training Loop Implementation

The multi-task training loop was perhaps the most intricate part of this project. I implemented:

1. **Dynamic task weighting**: I created a TaskWeightScheduler that adjusts the contribution of each task's loss based on their relative uncertainties, preventing one task from dominating training
2. **Proper gradient handling**: Including gradient clipping to prevent exploding gradients that can occur when tasks have conflicting objectives
3. **Comprehensive metrics tracking**: Beyond just accuracy, I track F1 scores and confusion matrices for both tasks to get a complete picture of model performance
4. **Early stopping**: To prevent overfitting, especially important with limited review data

One particularly interesting challenge was balancing the tasks. In Starbucks reviews, sentiment analysis is often easier than attribute classification (it's easier to tell if someone is happy than what they're specifically happy about). My uncertainty-based weighting approach helps balance this by giving more weight to the task that's struggling more.

### Streamlit Interface

As a bonus, I built a Streamlit interface that makes the project accessible to non-technical users. The interface allows:

- Live analysis of custom review inputs
- Visualization of model predictions with probability distributions
- Browsing sample reviews with model predictions
- Displaying model performance metrics and training curves


## How to Run the Project

### Option 1: Using Docker (Recommended)

```bash
# Build the Docker container
docker build -t starbucks-reviews-analysis .

# Run the container with Streamlit interface
docker run -p 8501:8501 starbucks-reviews-analysis

# Open your browser and navigate to:
# http://localhost:8501
```


### Option 2: Running Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --mode train --data_path starbucks_reviews.csv --epochs 5

# Run the Streamlit interface
streamlit run app.py
```


### Additional Command Options

```bash
# Evaluate a trained model
python main.py --mode evaluate --model_path models/starbucks_mtl_best.pt

# Make predictions on a specific review
python main.py --mode predict --review_text "The coffee was excellent but the ambiance was too noisy" --model_path models/starbucks_mtl_best.pt
```


## Challenges and Learnings

The most challenging aspect was working with the pre-labeled Starbucks reviews dataset, particularly:

1. **Handling class distributions**: Some attributes like "food" appeared more frequently than others, requiring careful stratification during dataset splitting
2. **Model capacity balancing**: Finding the right size for task-specific heads to prevent overfitting while maintaining expressive power
3. **Training time management**: BERT-based models take significant time to train, so I had to optimize my implementation for efficiency

Through this project, I've gained deeper insights into the nuances of multi-task learning, particularly how sharing representations can help models generalize better while keeping task-specific components allows for specialization. I've also learned how careful consideration of freezing strategies can dramatically impact model performance and training efficiency.

## Future Improvements

Given more time, I would:

1. Implement aspect-based sentiment analysis to detect sentiment toward specific attributes
2. Explore hierarchical classification for more fine-grained attribute categorization
3. Add cross-validation to improve reliability of performance metrics
4. Experiment with alternative backbone models like RoBERTa or ALBERT
5. Add store-specific analysis to identify location-based patterns in customer satisfaction

---

I hope this README provides a clear picture of my implementation approach and thought process. If you have any questions or need clarification on any aspect of the project, please don't hesitate to reach out!

