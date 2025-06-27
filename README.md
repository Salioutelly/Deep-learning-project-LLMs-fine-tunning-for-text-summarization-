# Deep-learning-project-LLMs-fine-tunning-for-text-summarization
# 🧠 Dialogue Summarization using SAMSum Dataset

## 🎯 Objective

The goal of this project is to build and evaluate transformer-based models for summarizing informal, human-like dialogues. Using the **SAMSum dataset**, which contains over 16,000 annotated chat dialogues, the project aims to generate concise and coherent summaries that preserve the essence of multi-turn conversations.

This work contributes to the growing field of **dialogue summarization**, where the challenge lies in capturing nuance, conversational flow, and contextual dependencies across multiple speaker turns.

## 📦 Dataset

- **Name**: SAMSum Dataset  
- **Size**: ~16,000 conversations  
- **Content**: Informal dialogues (chat-like messages) paired with corresponding human-written summaries  
- **Use Case**: Conversational summarization, improving AI's ability to distill multi-turn interactions into meaningful summaries.

## 🤖 Model Definitions

### 1. BART (Bidirectional and Auto-Regressive Transformers)
- **Architecture**: Sequence-to-sequence transformer with a bidirectional encoder and an autoregressive decoder.
- **Strengths**: Combines BERT-style contextual encoding with GPT-style text generation, making it powerful for generating coherent summaries.

### 2. T5 (Text-to-Text Transfer Transformer)
- **Architecture**: Treats all NLP tasks as text-to-text problems. It includes both encoder and decoder components.
- **Strengths**: Highly adaptable and effective for both extractive and abstractive summarization tasks. Easily customizable for different input/output formats.

## 📈 Metrics Definition

The following metrics were used to evaluate model performance on the dialogue summarization task:

| Metric        | Description |
|---------------|-------------|
| **Training Loss**    | Measures how well the model fits the training data. Lower values indicate better learning. |
| **Validation Loss**  | Helps detect overfitting. A big gap between training and validation loss suggests poor generalization. |
| **ROUGE-1**          | Overlap of individual words (unigrams) between generated and reference summaries. |
| **ROUGE-2**          | Overlap of bigrams (two consecutive words). Captures fluency and phrase coherence. |
| **ROUGE-L**          | Longest common subsequence between predicted and reference summaries. Reflects sequence fluency. |
| **ROUGE-LSum**       | A variation of ROUGE-L tailored for summarization tasks across sentence boundaries. |
| **Gen Len**          | Average length of generated summaries. Indicates verbosity or conciseness. |

## 🛠️ Tools & Libraries

- Python 3.x
- HuggingFace Transformers
- PyTorch / TensorFlow
- ROUGE scoring (via `evaluate` or `rouge-score`)
- Jupyter Notebook for experimentation

## 🔧 Training Details

- **Data preprocessing**: Dialogue-cleaning and tokenization
- **Train/Validation split**: Custom split or provided splits
- **Fine-tuning**: BART and T5 were fine-tuned on the SAMSum dataset
- **Hardware used**: GPU-enabled training environment (e.g., Google Colab or local CUDA setup)

