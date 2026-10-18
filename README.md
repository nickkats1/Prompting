<div align="center">

# Prompting & Language Models

**NLP fundamentals, Transformers, prompting, retrieval, and post-training.**

A collection of eight educational notebooks and a decoder-only Transformer implemented in PyTorch. The examples move from representing text to generating responses, retrieving context, and fine-tuning a pretrained language model.

[Notebooks](#notebooks) · [Quick start](#quick-start) · [Transformer](#transformer-from-scratch) · [Training results](#training-results)

</div>

## Overview

This project explores how language models represent text, predict tokens, and use instructions and external information. The notebooks combine explanations with executable examples and saved outputs.

- **Text representations:** Tokenization, TF-IDF, embeddings, and contextual BERT vectors.
- **Language modeling:** A character-level Transformer built from individual attention heads, feed-forward layers, and residual blocks.
- **Prompting and generation:** GPT-2 sampling, message roles, worked reasoning examples, and Qwen3 thinking mode.
- **Retrieval and tools:** BM25 and dense retrieval built directly with BM25S, Sentence Transformers, and FAISS; web-search tool calling through LangChain.
- **Post-training:** Supervised fine-tuning of Llama 3.2 1B, followed by Direct Preference Optimization (DPO).

## Notebooks

Open a topic directly or follow the sequence below. Saved outputs let you inspect the examples without downloading models or running training.

| | Notebook | What it covers |
| :---: | --- | --- |
| 01 | [NLP fundamentals](examples/nlp-intro.ipynb) | PDF extraction, tokenization, named entities, token IDs, TF-IDF, and random embeddings. |
| 02 | [BERT: words in context](examples/bert_example.ipynb) | Compare contextual vectors for **orange** as a fruit and a color using cosine similarity. |
| 03 | [A Transformer from scratch](examples/transformer.ipynb) | Build, train, and sample from a small character-level language model. |
| 04 | [GPT-2: generating text](examples/GPT2-prompting-example.ipynb) | Compare sampled continuations at two temperatures with the same prompt and seed. |
| 05 | [Prompting and thinking](examples/prompting.ipynb) | Message roles, standard and chain-of-thought examples, and Qwen3 thinking mode. |
| 06 | [Tool calling](examples/tool-calling.ipynb) | Execute a requested web search and return its results to the conversation. |
| 07 | [Retrieval-augmented generation](examples/rag-example.ipynb) | Compare BM25 and embedding search over *1984*, then generate an answer with passage references. |
| 08 | [Supervised fine-tuning and DPO](examples/sft-dpo.ipynb) | Train Llama 3.2 1B on instructions and preference pairs, evaluate, and save checkpoints. |

## Quick start

Start with the Transformer notebook. Its small defaults run on CPU and require no pretrained weights or API key.

```bash
git clone https://github.com/nickkats1/prompting-base-models.git
cd prompting-base-models
python3 -m venv venv
source venv/bin/activate
python -m pip install jupyterlab ipykernel torch pypdf matplotlib tqdm
python -m ipykernel install --user --name prompting-base-models --display-name "Prompting & Language Models"
cd examples
jupyter lab transformer.ipynb
```

Select **Prompting & Language Models** as the kernel and run cells in order. Launch Jupyter from `examples/`; the book examples use the included `data/orwell1984.pdf` in the repository root.

### Dependencies by notebook

Install the packages for the notebook you want to run in the same environment. These are dependency lists, not a pinned environment; there is no repository-wide requirements file.

| Notebook | Packages |
| --- | --- |
| Transformer | `torch pypdf matplotlib tqdm` |
| NLP fundamentals | `PyPDF2 nltk spacy scikit-learn pandas numpy` |
| BERT | `torch transformers pandas` |
| GPT-2 and prompting | `torch transformers accelerate` |
| Tool calling | `python-dotenv langchain-openai langchain-core langchain-community ddgs` |
| RAG | `torch transformers sentence-transformers faiss-cpu bm25s pypdf` |
| SFT and DPO | `torch transformers accelerate datasets trl` |

For NLP, also run `python -m spacy download en_core_web_sm`. The notebook downloads NLTK's `punkt_tab` resource. Jupyter supplies IPython for notebook displays.

### Hardware and access

| Examples | Requirements |
| --- | --- |
| Transformer | CPU; uses CUDA when available. |
| BERT, GPT-2, prompting, RAG | Hugging Face model downloads. CPU execution is possible; model downloads and generation can take time. |
| Llama SFT/DPO | A GPU supporting the notebook's BF16 and TF32 settings, sufficient memory for full-weight training, and Hugging Face access to Llama 3.2. |
| Tool calling | An OpenAI API key with available credits, plus internet access for search. |

For tool calling, create `.env` in the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

`.env` is excluded by `.gitignore`. The example uses GPT-4o through LangChain and executes one round of requested web-search calls.

## Transformer from scratch

The [walkthrough](examples/transformer.ipynb) connects each model component to its implementation in [transformer/](transformer/).

```text
Character IDs
     │
Token embeddings + learned position embeddings
     │
┌────────────────────────────────────────────┐
│ LayerNorm → Causal attention → Residual add │
│ LayerNorm → Feed-forward MLP → Residual add │ × N blocks
└────────────────────────────────────────────┘
     │
Final LayerNorm → Vocabulary logits → Next character
```

The default model has **66,424 parameters**: two attention heads, 48 embedding dimensions, two blocks, and a 32-character context. Training uses batches of 16 and stops after 300 updates or approximately 60 seconds. Validation samples 128 windows spread across the final 10% of the text, separated from training text before constructing windows.

| File | Implementation |
| --- | --- |
| [attention.py](transformer/attention.py) | Per-head query, key, and value projections; scaled dot-product attention; causal masking. |
| [mlp.py](transformer/mlp.py) | Two-layer feed-forward network with ReLU. |
| [transformer_block.py](transformer/transformer_block.py) | Pre-normalization and residual connections. |
| [transformer.py](transformer/transformer.py) | Embeddings, stacked blocks, vocabulary logits, cross-entropy loss, and generation. |
| [sampling.py](transformer/sampling.py) | Temperature, top-k, top-p, and min-p filtering. |
| [utils.py](transformer/utils.py) | Character tokenizer and shifted next-token windows. |

The notebook’s CPU run reduces sampled validation loss from **4.653 to 2.638** after 300 updates. Generated text remains rough at this training budget. Increase `max_steps` and `max_train_seconds` to experiment with longer runs.

## Retrieval and tool use

The [RAG notebook](examples/rag-example.ipynb) splits the extracted book into passages of up to 150 words. BM25 searches by lexical relevance; Sentence Transformers embeds the passages, and FAISS searches normalized vectors using inner product. Both retrievers receive the same question so their results can be compared.

The retrieved passages are supplied to **Qwen2.5-0.5B-Instruct** with their original passage IDs. The answer and source passages are displayed together. Passage length is set by word count; when changing the source text, check token lengths against the embedding model’s input limit.

The [tool-calling notebook](examples/tool-calling.ipynb) demonstrates another workflow: the model requests a search, Python executes it, and the results return as tool messages before the final response. Search results and generated citations still need to be checked.

## Training results

### Llama 3.2 1B: SFT → DPO

The [post-training notebook](examples/sft-dpo.ipynb) starts with a pretrained base model, trains on Alpaca Cleaned instructions, saves the SFT checkpoint, then trains on Orca chosen/rejected response pairs. DPO uses a frozen copy of the SFT checkpoint as its reference model.

Both stages use a 90/10 training–validation split, a 512-token sequence limit, and 100 optimizer updates. Validation and checkpoint saving run every 50 updates; each trainer restores the checkpoint with the lowest validation loss.

| Stage | Dataset | Validation loss: update 50 | Validation loss: update 100 |
| --- | --- | ---: | ---: |
| SFT | `yahma/alpaca-cleaned` | 1.455908 | 1.328105 |
| DPO | `Intel/orca_dpo_pairs` | 0.061847 | 0.052318 |

These values come from saved Colab outputs. SFT and DPO optimize different objectives, so their losses are not directly comparable. A matched greedy-generation example shows the base model repeating the prompt, SFT producing a relevant numbered response, and DPO changing its wording. That single example does not establish an overall quality improvement from DPO.

The training notebook saves under `/content/` in its Colab runtime. Download checkpoints separately to keep the weights; the notebook preserves code and outputs.

## Scope and limitations

These are educational examples and short training runs. The Transformer learns character-level patterns from *1984*; it is not a pretrained assistant. BERT similarities, retrieval results, and individual generated responses illustrate model behavior rather than establish benchmark performance.

The prompting notebook compares standard and chain-of-thought examples using Qwen2.5-0.5B, then demonstrates Qwen3-0.6B thinking mode. It does not reproduce the experiments in the original chain-of-thought paper.

The SFT/DPO metrics are saved Colab results. Running those cells again requires suitable GPU hardware and model access. The project has no maintained automated test suite or pinned dependency environment, so package changes may require adjustments.

## Repository layout

```text
examples/              Eight notebooks with explanations and saved outputs
transformer/           PyTorch model, sampling, and data utilities
tools/search.py        Web-search tool
data/orwell1984.pdf    Source text for PDF and book examples
assets/images/         Notebook illustrations
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## License

Code is available under the [MIT License](LICENSE). Models, datasets, papers, and book text retain their respective licenses and access conditions.
