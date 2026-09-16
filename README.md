# Prompting

This repository consist of various tasks with LLM's starting from 2019 with GPT2 and includes: SFT, CoT, Reasoning Models, GRPO and tool calling.

## Guide

If you have ever used chat gpt or claude or any of the commercial products, this is the same thing, except the models have not yet had instruction tuning done on them yet (I will do that in this notebook as well). This will show that when that is not done, the model is just predicting the next token based on all tokens that came before it. This also covers SFT. Which is a way for the model to have completions that are not just completing the prompt.


## Getting Started

Make sure to have a .env file and hide it if you ever commit with it. You need a Hugging Face API key.

```bash
touch .env
cat > .env "your api key"
```

## What is covered in this repo

This repo covers various techniques with modern nlp and Transformer-based LLMs including the following:

- **Elementary NLP Tasks**: [nlp-intro](examples/nlp-intro.ipynb)
- **Bert**: [bert](examples/bert_example.ipynb)
- **GPT2-base-model**: [gpt2](examples/GPT2-prompting-example.ipynb)
- **Llama-instruct**: [llama-instruct]
- **Types of prompting and types of prompts**: [prompting](prompting.ipynb)
- **Tool Calling**: [tools](tools.ipynb)
- **Supervised Fine-Tuning and LoRA**: [gpt2-instruct](examples/SFT.ipynb)

**Note**: This is not tuned well. I did SFT on a `GTX-1080` gpu and could only run a single epoch. 

## Requirements

You should have access to a gpu for torch and a hugging face api key which is free at the following: [huggingface](https://huggingfacehub.com). If you do not have access to a gpu, use Google Colab and select anything that is not "cpu". 


## Why this repo

`AI` is currently the most over-used word in the world right now and I am actually surprised about sheer amount of people who know literally nothing about it and view ChatGPT as all-knowing god. The word `AI` is a umbrella terms that means a lot of things. Usually the general public, means a Large Language model when they talk about `ai` or some variant of the transformer (with vision, audio ect) Image and audio transformers will not be covered in this repo. 

## The Transformer

The transformer came out in 2017's landmark paper "Attention is All you Need". This introduces a architecture with a encoder block and a decoder block. The decoder-only autoregressive part is what most LLM's are trained on. The decoder-only part, only predicts the next token: [example](examples/GPT2-prompting-example.ipynb), [transformer](transformer.ipynb)




## License


License[MIT](LICENSE)





