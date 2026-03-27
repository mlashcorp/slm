# 2 OLMo 2 Furious

**arXiv:** 2501.00656  
**Submitted:** 31 Dec 2024 (v1), last revised 8 Oct 2025 (v3)  
**Authors:** Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, and 42 other authors  
**URL:** https://arxiv.org/abs/2501.00656  
**PDF:** https://arxiv.org/pdf/2501.00656

## Abstract

We present OLMo 2, the next generation of our fully open language models. OLMo 2 includes a family of dense autoregressive language models at 7B, 13B and 32B scales with fully released artifacts -- model weights, full training data, training code and recipes, training logs and thousands of intermediate checkpoints.

In this work, we describe our modified model architecture and training recipe, focusing on techniques for achieving better training stability and improved per-token efficiency. Our updated pretraining data mixture introduces a new, specialized data mix called Dolmino Mix 1124, which significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training (i.e. specialized data during the annealing phase of pretraining).

Finally, we incorporate best practices from Tülu 3 to develop OLMo 2-Instruct, focusing on permissive data and extending our final-stage reinforcement learning with verifiable rewards (RLVR).

Our OLMo 2 base models sit at the Pareto frontier of performance to training compute, often matching or outperforming open-weight only models like Llama 3.1, Qwen 2.5, and Gemma 2 while using fewer FLOPs and with fully transparent training data, code, and recipe. Our fully open OLMo 2-Instruct models are competitive with open-weight only models of comparable size and even some proprietary models like GPT-3.5 Turbo and GPT 4o Mini.

## Key Points

- **Model sizes:** 7B, 13B, and 32B parameters
- **Architecture:** Dense autoregressive language models
- **Training improvements:** Better stability, improved per-token efficiency
- **Data mixture:** Dolmino Mix 1124 for late-stage curriculum training (annealing phase)
- **Post-training:** RLVR (Reinforcement Learning with Verifiable Rewards) from Tülu 3
- **Performance:** Pareto frontier - matches/outperforms Llama 3.1, Qwen 2.5, Gemma 2 with fewer FLOPs

## Two-Stage Training

1. **Stage 1:** Large-scale pretraining on web-based data (OLMo-mix-1124)
2. **Stage 2:** High-quality, domain-focused continuation (Dolmino-mix-1124)

## Comments

Shorter version accepted to COLM 2025. Updated to include 32B results.

Model demo available at: http://playground.allenai.org

## Subjects

- Computation and Language (cs.CL)
- Machine Learning (cs.LG)

## Citation

```bibtex
@misc{olmo20242olmo2furious,
  title={2 OLMo 2 Furious},
  author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Allyson Ettinger and Michal Guerquin and David Heineman and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Jake Poznanski and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
  year={2024},
  eprint={2501.00656},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.00656}
}
```
