## Temperature Scaling

Changing the distribution for sampling, and knowledge distillation. 
if we use q/T where q is the logits and T-> 0 then the distribution will be more peaked. 


A higher temperature results in a more uniform probability distribution, encouraging the model to make less obvious, more diverse word choices. A lower temperature makes the distribution peakier, favoring more likely words and generating more predictable text.

Context: During the decoding process in language models, temperature scaling can influence the generation of text.
Usefulness: Adjusting the temperature during decoding can help balance between creativity and coherence in the generated text. For instance, in dialogue systems, a moderate temperature might help generate responses that are both relevant and varied, improving the quality of the interaction.


## Roberta

Different objective functions and different embeddings. There was some derivations in the GLOVE paper which showed GLOVE and Word2Vec are very similar.

BERT is trained on two primary tasks:

Masked Language Model (MLM): Randomly masks some of the tokens from the input, and the objective is to predict the original token based only on its context. Unlike traditional language models, this allows BERT to learn a bidirectional representation of the text.

Next Sentence Prediction (NSP): Given two sentences A and B, the model predicts whether B is the actual sentence that follows A in the original text. This helps BERT understand the relationship between sentences.


Roberta:

- Eliminating NSP: RoBERTa argues that the NSP task is not as crucial as previously thought. It removes this objective, focusing solely on the MLM task

Training on Larger Batches and Data: It is trained on much larger batches and datasets, which helps in learning more generalizable representations.
Longer Training: It undergoes longer training with more iterations, contributing to its robustness.
Dynamic Masking: Unlike BERT, which statically masks words during pre-processing, RoBERTa applies dynamic masking, where the masking pattern changes during the training rounds.
Byte-Pair Encoding (BPE): RoBERTa uses a byte-level BPE as a tokenizer which allows better handling of out-of-vocabulary words.

## Attention

The attention mechanism in LSTMs computes a set of attention weights.
These weights are applied to the hidden states of the LSTM encoder to create a context vector.
The context vector is then combined with the current hidden state of the LSTM decoder to generate the next output.

## Self-attention

Parallelization: Unlike RNNs, the self-attention mechanism can process all tokens in a sequence in parallel, significantly speeding up computation.

Context Awareness: It captures the context of each token from the entire sequence, making it powerful for understanding the meaning in context.

Long-range Dependencies: Self-attention can manage long-range dependencies in text better than many traditional models.

### Advantage:

Self-attention is inherently bidirectional (or non-directional) in nature, meaning it simultaneously considers past and future context in sequence modeling.
Unlike LSTM attention, self-attention does not rely on recurrent connections, allowing for greater parallelization and efficiency in processing long sequences.

#### steps:

input:

Input Sequence: Begin with a sequence of input tokens (words or subwords), typically represented as embeddings. These embeddings are dense vector representations of the tokens.
Positional Encoding: To preserve the order of tokens in the sequence, positional encodings are added to the input embeddings. This is essential since self-attention by itself doesn't take the order of tokens into account.

2. Generate Query, Key, and Value Vectors
For each token in the input sequence, generate three different vectors through linear transformations: the Query vector (Q), Key vector (K), and Value vector (V). These are created using separate weight matrices for queries, keys, and values that are learned during training.

attn score:

Dot Product of Q and K: For each token, calculate the dot product between its Query vector and the Key vectors of all tokens in the sequence. This operation measures the compatibility or relevance of other tokens with respect to the current token.
Scaling: Scale the dot products by a factor (usually the square root of the dimension of key vectors) to avoid extremely small gradients during training.

softmax:
Softmax Function: Apply the softmax function to the scaled dot products for each token. This step converts the scores into probabilities, ensuring they sum to 1. The softmax score determines how much focus (or attention) each token should get in relation to the current token.

Multiply Scores with Value Vectors
Weighted Values: Multiply the softmax probabilities with the corresponding Value vectors. This step essentially retrieves the most relevant information from each part of the input sequence, as indicated by the calculated attention scores.

## Glove

#### Steps
Create a Co-Occurrence Matrix: For a given corpus, a matrix is created where each element Xij represents how often word i appears in the context of word j.

Define a Weighting Function: A weighting function is used to decrease the influence of rare and frequent co-occurrences. This helps to balance the focus on relevant versus less relevant word associations.

Minimize the Cost Function: The core of GloVe is a learning process that tries to minimize the difference between the dot product of the vector representations of two words and the logarithm of their co-occurrence probability. The cost function incorporates the weighting function and works towards making the word vectors more representative of the actual meanings.


#### Advantages

Combines Count-Based and Predictive Models: It leverages the benefits of both global matrix factorization (as seen in count-based methods like LSA) and local context window methods (like Word2Vec), providing a comprehensive semantic capture.

Efficiently Captures Word Analogies: It can represent linear relationships between words, making it effective in solving word analogy tasks (like "man" is to "woman" as "king" is to "queen").

Scalability: GloVe can be trained on large corpora efficiently, making it suitable for tasks that require understanding of nuanced and complex language semantics.


## LSTM

Contextual Understanding: LSTM networks can capture the order and context in which words appear in a sentence. This is crucial in understanding the nuances of language, as the meaning of words can change dramatically based on their context. For instance, in movie reviews, phrases like "not good" and "good" have opposite meanings, which LSTMs can distinguish but BoW models might struggle with.

Handling Long Dependencies: LSTMs are designed to remember information over long sequences, making them suitable for handling sentences or paragraphs where context from the beginning might influence the meaning at the end.


## ULM-FIT

### Steps:

Pre-Training Phase
Language Model Training: ULM-FiT starts with a pre-trained language model. This model is usually trained on a large, general-purpose corpus of text (such as Wikipedia articles) to learn the structure and nuances of the language.

Learned Representations: The pre-trained model captures general features of the language, including syntax, grammar, and basic word relationships, without any specific knowledge about the domain of movie reviews.

Fine-Tuning Phase
Domain-Specific Training: The pre-trained language model is then fine-tuned on a corpus more specific to the target task, which in this case would be a dataset of movie reviews. This step helps the model learn about language usage specific to movie reviews.

Gradual Unfreezing: ULM-FiT introduces a novel technique called "gradual unfreezing". Instead of fine-tuning all layers of the model at once, you first fine-tune the last layers and gradually unfreeze earlier layers. This prevents catastrophic forgetting, where the model forgets what it learned during pre-training.

Classifier Training
Adding a Classifier: After fine-tuning the language model on movie reviews, a classification layer is added on top of the model. This layer will make the actual predictions (e.g., genres or sentiment of the movie).

Fine-Tuning for Classification: Finally, the entire model (language model + classifier) is fine-tuned on the classification task. Here again, gradual unfreezing and discriminative learning rates (different learning rates for different layers) are used to optimize performance.

### Advantages:

Transfer Learning: ULM-FiT leverages a pre-trained language model, which provides a strong foundation of language understanding. This is different from training an LSTM from scratch, where the model learns only from the task-specific data.

Efficiency and Effectiveness: Fine-tuning a pre-trained model is generally more efficient and can lead to better performance, especially when the task-specific dataset is small. The model can leverage its prior knowledge about language, which a freshly trained LSTM wouldn't have.

Advanced Fine-Tuning Techniques: Techniques like gradual unfreezing and discriminative learning rates are unique to ULM-FiT and help in retaining the pre-trained knowledge while adapting to the new task.

## GPT

### advantage
Autoregressive Language Modeling: GPT (Generative Pretrained Transformer) models are autoregressive, meaning they predict the probability of a sequence by conditioning on the previous tokens. This leads to a deep understanding of language structure and context.

Powerful Pre-training: Like ULM-FiT, GPT models are also pre-trained on a large corpus, allowing them to learn a rich understanding of language. However, GPT models often have a larger and more diverse pre-training dataset, leading to potentially more robust language representations.

Less Need for Task-Specific Adjustments: GPT models can be fine-tuned for specific tasks with minimal architecture changes. The same model architecture used for pre-training is often used for fine-tuning, which can simplify the process.

State-of-the-Art Performance: GPT and similar transformer-based models have demonstrated state-of-the-art performance in a variety of NLP tasks, including classification.

Highly Scalable Architecture: The transformer architecture in GPT scales well with increased data and compute resources, often leading to improved performance with larger models and datasets.

### problem：
Computational Resources: GPT models, especially larger versions, require significant computational resources for both pre-training and fine-tuning. This can be a barrier for researchers or practitioners with limited access to high-powered computing.

Data and Energy Intensive: The pre-training process is data and energy-intensive, raising concerns about environmental impact and the feasibility for continuous re-training.

Potential Overfitting on Smaller Datasets: While GPT models excel with large datasets, they can sometimes overfit on smaller, task-specific datasets if not fine-tuned carefully.

Complexity in Interpretability: Understanding why a GPT model makes a certain prediction can be more challenging due to the complexity and the large number of parameters in the model.

Dependency on Pre-trained Knowledge: GPT models heavily rely on the knowledge acquired during pre-training. If the pre-training data is biased or limited in certain aspects, these issues can propagate to the fine-tuned model.


## BERT


###  Features

Bidirectional Context Understanding: BERT is designed to understand the context of a word based on all of its surroundings (left and right of the word). This makes it particularly strong in understanding the full context of a sentence, which is crucial in text classification.

Fine-Tuning for Specific Tasks: BERT can be fine-tuned with additional output layers for specific tasks, including text classification. This makes it adaptable and effective for specialized domains.

Pre-training on Language Understanding: BERT's pre-training involves predicting missing words in a sentence, which gives it a strong base in language understanding.


### Comparison

For tasks where the understanding of the complete context of a sentence is crucial, BERT might have an edge due to its bidirectional nature.

GPT models, while powerful in language generation and adaptable to various tasks, might not always match BERT's performance in specific text classification scenarios where the full context is key.

The choice between BERT and GPT depends on the specific requirements of the text classification task, including the nature of the data, the importance of context, and the desired outcomes.


### Beam Search
Branching:

At each step, the algorithm explores all possible next steps (or expansions) from the current state. In language tasks, these are often the possible next words in a sequence.
Beam Width:

The key feature of beam search is the 'beam width' (denoted as 'k'). This parameter limits the number of branches that are kept at each step. Only the 'k' most promising branches (according to a scoring function) are retained for further exploration.
The scoring function often involves probabilities assigned to sequences, especially in models like neural networks.
Pruning:

Paths that are not in the top 'k' are pruned or discarded, which significantly reduces the memory requirements compared to a full breadth-first search.


### CRF and MEmm
Context Sensitivity: CRFs are generally better at capturing dependencies across the entire input sequence due to their global normalization, whereas MEMMs may struggle with dependencies that span over multiple steps.
Label Bias: CRFs do not suffer from the label bias problem as much as MEMMs, making them more balanced in their predictions.
Complexity and Computation: CRFs can be more computationally intensive to train and use, as they consider the entire sequence for their calculations.