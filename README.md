# Data-Scientist-Ruchitha-anugu-Set-3

PART A: Multiple Choice Questions

Q1.  In a production RAG system, re-ranking retrieved chunks before passing them to the LLM is beneficial because:
A)  It reduces the embedding dimensionality of the retrieved vectors
B)  Initial ANN retrieval optimises for approximate similarity; a cross-encoder re-ranker can more precisely score query-document relevance, improving final answer quality
C)  Re-ranking updates the vector store index with the new query embedding
D)  It allows the LLM to attend to all documents in the corpus without a retrieval step
**ANSWER- B**

Q2.  "Semantic chunking" in a RAG pipeline differs from fixed-size chunking because it:
A)  Uses a fixed token limit but adds an extra overlap of 50 tokens between chunks
B)  Dynamically determines chunk boundaries by detecting semantic shifts in the text (e.g., using embedding similarity between consecutive sentences) rather than cutting at a fixed token count
C)  Splits documents only at paragraph-level HTML tags
D)  Embeds the entire document as a single chunk regardless of length
**ANSWER- B**

Q3.  An LLM agent using a "plan-and-execute" pattern differs from a standard ReAct agent in that:
A)  It calls tools in a random order to explore the action space
B)  It first generates a complete multi-step plan, then executes each step sequentially using sub-agents or tools, separating the planning and execution phases
C)  It generates only a single action before returning the answer to the user
D)  It uses reinforcement learning to update its policy after every tool call
**ANSWER- B**

Q4.  Sentence-BERT (SBERT) adapts BERT for semantic similarity by:
A)  Adding a decoder stack so BERT can generate sentences
B)  Using a Siamese / twin-network structure where two sentences are encoded independently and their embeddings are compared with cosine similarity, trained on NLI or STS data
C)  Averaging BERT's token embeddings without any fine-tuning
D)  Replacing BERT's attention with a recurrent layer for faster pair comparison
**ANSWER- B**

Q5.  The key reason encoder-only models are preferred for token classification tasks (e.g., NER) over decoder-only models is:
A)  Encoder-only models have more parameters and are thus more expressive
B)  Bidirectional context allows every token's representation to incorporate information from both preceding and following tokens, which is crucial for accurate labelling of each position
C)  Decoder-only models cannot process sequences longer than 512 tokens
D)  Encoder-only models natively output one label per token without any modifications
**ANSWER- B**

Q6.  In a decoder-only model, the feed-forward network (FFN) sublayer that follows multi-head attention typically:
A)  Applies attention between the current token and all other tokens again for deeper contextualisation
B)  Applies two linear transformations with a non-linearity (e.g., GELU or SwiGLU) independently and identically to each position - functioning as a position-wise learned feature transformation
C)  Reduces the hidden dimension to the vocabulary size for next-token prediction at every layer
D)  Stores long-term memory by writing to an external key-value memory module
**ANSWER- B**

Q7.  Relative position encodings (e.g., ALiBi) differ from absolute positional embeddings in that:
A)  Relative encodings are added to token embeddings before the first layer while absolute encodings are added inside each attention layer
B)  ALiBi adds a fixed, non-learned linear bias to attention scores based on the distance between query and key positions, allowing the model to attend over lengths beyond those seen during training
C)  Relative encodings use trainable parameters while absolute encodings are fixed sinusoids
D)  ALiBi encodes position only in the first and last layers to reduce computation
**ANSWER-  B**




Q8.  DeepSpeed ZeRO Stage 3 reduces GPU memory in distributed training by:
A)  Quantizing the model to INT8 before distributing it across GPUs
B)  Partitioning optimizer states, gradients, AND model parameters across all GPUs so no single GPU holds a full copy, with communication collectives reconstructing shards on demand
C)  Using pipeline parallelism to place different layers on different GPUs
D)  Pruning 50% of model weights before initiating distributed training
**ANSWER-  B**


Q9.  When serving a large language model with PagedAttention (used in vLLM), memory fragmentation is reduced by:
A)  Sorting requests by output length so that KV-cache blocks are allocated sequentially
B)  Managing KV-cache in fixed-size non-contiguous "pages" mapped dynamically to logical sequence positions, similar to OS virtual memory paging, preventing fragmentation from variable-length sequences
C)  Pre-allocating a fixed KV-cache block per token position at server startup
D)  Offloading completed sequence KV-caches to CPU RAM to free GPU memory
**ANSWER-  B**


Q10.  The primary motivation for using BF16 over FP16 in training large language models is:
A)  BF16 has higher precision (more mantissa bits) than FP16, reducing rounding error in activations
B)  BF16 has the same 8-bit exponent range as FP32, making it less prone to overflow/underflow during training, while its 7-bit mantissa provides sufficient precision for gradient flow
C)  BF16 is a hardware standard that runs on CPUs but not GPUs
D)  BF16 consumes half the memory of FP16, enabling twice the batch size
**ANSWER- B**


Q11.  Adapter-based fine-tuning (Houlsby adapters) inserts small bottleneck modules into each transformer layer. The key design choice is:
A)  Adapters replace the self-attention layer entirely with a lightweight MLP
B)  A down-projection reduces the hidden dimension to a bottleneck size r, a non-linearity is applied, and an up-projection restores the original dimension - with a residual skip connection ensuring near-identity initialisation
C)  Adapters are inserted only after the final transformer layer to modify the output distribution
D)  Adapters share weights across all transformer layers to minimise parameter count
**ANSWER- B**


Q12.  DPO (Direct Preference Optimization) is an alternative to RLHF-PPO that:
A)  Uses a reward model to score responses and updates the policy with proximal policy optimization
B)  Directly optimises the LLM on human preference pairs (chosen vs rejected) using a closed-form re-parameterization of the RL objective, eliminating the need for a separate reward model and RL training loop
C)  Generates preference data synthetically using a larger teacher model and distils it into the student
D)  Fine-tunes only the embedding layers on preference data, leaving the transformer weights frozen
**ANSWER- B**


Q13.  SmoothQuant addresses the challenge of quantizing LLMs with activation outliers by:
A)  Clipping all activation values above a fixed threshold before quantization
B)  Mathematically migrating quantization difficulty from activations to weights by multiplying a per-channel smoothing factor into the weights and dividing it from the activations, making both easier to quantize
C)  Replacing INT8 activation quantization with FP8 to accommodate outliers
D)  Dropping outlier tokens from the attention computation before quantization
**ANSWER- B**


Q14.  GGUF (formerly GGML) quantization formats used by llama.cpp differ from GPTQ primarily in that:
A)  GGUF quantizes activations while GPTQ only quantizes weights
B)  GGUF stores quantized weights in a CPU-friendly format optimised for inference on consumer hardware (CPU/hybrid CPU-GPU), while GPTQ targets GPU-only inference with a specific dequantization kernel
C)  GGUF uses 8-bit quantization exclusively, while GPTQ supports 2-bit through 8-bit
D)  GPTQ is a file format while GGUF is a quantization algorithm
**ANSWER- B**


Q15.  The sacreBLEU library is important in MT research because:
A)  It computes a new neural BLEU variant that correlates better with human judgments
B)  It provides a standardised, reproducible BLEU implementation with defined tokenization rules, eliminating inconsistencies across different implementations that made scores incomparable across papers
C)  It combines BLEU with BERTScore into a single composite metric
D)  It is a web service that crowdsources human evaluations for MT outputs
**ANSWER- B**


Q16.  BERTScore evaluates generated text by:
A)  Fine-tuning BERT on the test set and computing its accuracy on held-out samples
B)  Computing token-level cosine similarity between contextual embeddings of hypothesis and reference tokens from a pretrained BERT-like model, using maximum similarity per token to compute precision, recall, and F1
C)  Measuring the edit distance between the BERT tokenizations of hypothesis and reference
D)  Using BERT's next-sentence prediction score to judge if the hypothesis follows from the reference
**ANSWER- A**

Q17.  For evaluating translation of domain-specific text (e.g., medical or legal), automatic metrics like BLEU are often insufficient because:
A)  BLEU was not designed for texts longer than 50 tokens
B)  Domain-specific translations may require exact technical terminology; BLEU's n-gram overlap cannot distinguish a wrong technical term from a synonym, and no standard domain-specific reference set may exist
C)  Legal and medical texts always contain numbers that BLEU ignores entirely
D)  Automatic metrics only work for common language pairs like English-French
**ANSWER- C**


Q18.  Mixture of Experts (MoE) architecture improves parameter efficiency in LLMs by:
A)  Training multiple small models and averaging their outputs at inference time
B)  Replacing each dense FFN layer with multiple expert FFN layers, using a router to activate only a sparse subset of experts (e.g., top-2) per token, so total parameters are large but FLOPs per token remain manageable
C)  Assigning different transformer heads to different tasks and routing inputs based on task type
D)  Using knowledge distillation from multiple teacher models into one student model
**ANSWER- B**


Q19.  Beam search during text generation differs from greedy decoding in that:
A)  Greedy decoding samples from the full probability distribution while beam search takes the argmax
B)  Beam search maintains K candidate sequences at each step, expanding all of them and keeping the top-K by cumulative log-probability, trading compute for higher quality compared to greedy's single-path search
C)  Beam search is stochastic; greedy decoding is deterministic
D)  Beam search runs multiple models in parallel; greedy uses one model
**ANSWER- D**


Q20.  In a RAG pipeline evaluation framework like RAGAS, "answer relevance" measures:
A)  Whether the retrieved context is topically related to the query
B)  Whether the generated answer directly and completely addresses the user's question - independently of whether the answer is factually grounded in the context
C)  The BLEU score between the generated answer and a gold reference answer
D)  The cosine similarity between the query embedding and the answer embedding
**ANSWER- A**





