# Transformer

## Summary
This is Chat Bot transformer model. This model's architecture and baseline code were taken from [Andrej Karpathy's video](https://youtu.be/kCc8FmEb1nY?feature=shared), which was initially trained on the MiniShakespeare dataset. I pretrained the model on several classic books, including The Count of Monte Cristo by Alexandre Dumas and The Brothers Karamazov by Fyodor Dostoevsky. I then took the pretrained model and trained it on the DailyDialog dataset, placing "A:" and "B:" tokens before respective speakers. The chatbot.py file can be run to have a conversation with the final product.

## Training Details  
- **Batch Size**: 32  
- **Epochs**: N/A (data was sampled randomly)  
- **Optimizer**: AdamW  
- **Learning Rates**:  
  - Pretraining: 1e-3  
  - Fine-tuning: 6e-5  
- **Regularization**: Dropout with a rate of 0.2
- **Context Length**: 256

## Datasets and Preprocessing  
- **Preprocessing**: Used Byte Pair Encoding (BPE) with a vocabulary size of 10,000.  
- **Dataset Splits**:  
  - Training set: First 80% of each book.  
  - Test set: Last 20% of each book.
 - **Dataset Sizes**:
   -Pretraining: ~4 million tokens
   -Fine-Tuning: ~144,000 tokens

## Evaluation and Metrics  
- **Evaluation Metric**: Cross-Entropy Loss.  

## Model Details and Behavior  
- Most sentences the model produces are syntactically correct, but it has trouble maintaining cohesion from sentence to sentence. Both the model and training set were small, however, so this was to be expected. Nonetheless, the model performed relatively well and is capable of keeping an amusing conversation.
- Due to the nature of the training set, the model speaks conversationally but is incapable of following instructions like modern LLMs are able to.
