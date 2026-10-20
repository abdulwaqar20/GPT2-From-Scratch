# GPT-2 From Scratch (NumPy)

A GPT-2 style decoder-only transformer built from scratch in NumPy, with no deep learning framework. Every component, including the backward pass, is written by hand.

## What's implemented
- Character-level tokenizer (encode/decode)
- Token and learned positional embeddings
- LayerNorm (forward and backward)
- Masked (causal) self-attention and multi-head attention
- Feed-forward network, residual connections, 4 stacked transformer blocks
- Tied input/output embeddings
- Softmax, cross-entropy loss, full manual backpropagation
- AdamW optimizer
- Autoregressive generation with temperature and greedy/sampling modes

## Config
d_model 32 | 4 heads | 4 layers | FFN 128 | context 32 | batch 16 | lr 6e-4 | 2000 steps

## Results
Loss dropped from 3.29 (step 0) to about 0.13 (step 1900) on a 430-character Hamlet excerpt.

Sample (prompt "the", temperature 0.8):
`the is the question, whether tis tis a noble r minob`

## Limitations (please read)
- The training text is tiny (430 characters, vocab 27), so the model memorises it instead of learning general language.
- No validation split, so the low loss shows fit, not generalisation.
- Smaller than real GPT-2 (ReLU instead of GELU, character-level tokens).
- This is an educational project to understand how transformers work.

## Run
pip install -r requirements.txt, then open the notebook and run all cells.

See `docs/report.docx` for a detailed write-up.