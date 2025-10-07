# 💬 Offline Chat Reply Recommendation System using Transformers

This project builds an **offline chat-reply recommendation system** that predicts the next possible reply from *User A* when *User B* sends a message — using previous conversation history as context.

---

## 🚀 Project Overview

The system uses **Transformer-based models (e.g., BERT, GPT, or T5)** fine-tuned on two-person conversation datasets to generate or recommend contextually appropriate replies.

### 🧠 Objective
- Train a transformer model to predict **User A’s next message** based on **User B’s previous message** and conversation history.
- Handle **long conversational data** efficiently using tokenization and attention mechanisms.
- Build an **offline inference setup** (no API calls) for real-time chat assistance.

---

## 🗂️ Dataset

You need two datasets containing long two-person chat conversations.  
Each dataset should follow a format similar to:

| User | Message | Timestamp |
|------|----------|------------|
| A | Hi, how are you? | 10:01 |
| B | I’m fine, you? | 10:02 |
| A | Doing great! | 10:03 |

> Preprocessing and tokenization scripts handle cleaning, truncation, and context formatting.

---

## ⚙️ Model Workflow

1. **Preprocessing:**
   - Tokenize chat history
   - Create context–response pairs
   - Limit sequence length using sliding windows

2. **Model:**
   - Base Transformer: BERT / DistilGPT2 / T5
   - Fine-tuned on conversation pairs
   - Loss: Cross-Entropy or Causal LM Loss

3. **Inference:**
   - Takes the last few turns of conversation
   - Predicts top candidate replies

---

## 🧰 Technologies Used

- Python
- PyTorch / TensorFlow
- Transformers (Hugging Face)
- NLTK / SpaCy
- Pandas, NumPy
- GPU/Colab for training

---

## 💻 Setup Instructions

```bash
# Clone this repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python preprocess_data.py

# Train the model
python train_model.py

# Generate replies
python predict.py
