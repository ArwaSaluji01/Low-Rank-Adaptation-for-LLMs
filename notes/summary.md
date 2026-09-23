# Project Summary: LoRA Fine-Tuning of DistilGPT2

## Overview

This project is an independent implementation and experimental study of **Low-Rank Adaptation (LoRA)**, a parameter-efficient fine-tuning method introduced in the paper *LoRA: Low-Rank Adaptation of Large Language Models*.

The objective was to understand how low-rank trainable matrices can be integrated into a pretrained Transformer model while keeping the original model weights frozen.

The implementation uses PyTorch and Hugging Face Transformers to adapt DistilGPT2 to instruction-formatted examples from the Databricks Dolly 15k dataset.

## Technical Approach

The project was developed incrementally, beginning with a standalone LoRA linear layer and progressing toward integration with the attention projection layers of DistilGPT2.

The main implementation stages included:

1. Loading and preparing the instruction-following dataset.
2. Evaluating the pretrained DistilGPT2 baseline.
3. Implementing a low-rank adaptation layer from scratch.
4. Integrating the LoRA layer into the GPT-2-style `c_attn` modules.
5. Freezing the pretrained model weights.
6. Verifying trainable parameters and gradient flow.
7. Fine-tuning the model using causal language modeling loss.
8. Comparing the baseline and LoRA-fine-tuned models.
9. Visualizing training progress.

The LoRA configuration used a rank of 4 and an alpha value of 4.

## Research Connection

The implementation is based on the core principle presented in the original LoRA paper: freeze pretrained weights and learn low-rank updates for adaptation.

However, this project is not an exact reproduction of the paper's experiments. The implementation uses DistilGPT2, a subset of the Dolly dataset, and the combined GPT-2-style `c_attn` projection rather than reproducing every target-module and evaluation configuration from the paper.

## Experimental Results

The final quantitative results will be added after the training and evaluation process is complete.

| Metric                         | Result |
| ------------------------------ | ------ |
| Baseline validation loss       | TBD    |
| LoRA validation loss           | TBD    |
| Baseline perplexity            | TBD    |
| LoRA perplexity                | TBD    |
| Total parameters               | TBD    |
| Trainable parameters           | TBD    |
| Trainable parameter percentage | TBD    |

Qualitative evaluation will examine response relevance, coherence, repetition, completeness, and instruction-following behavior.

## Learning Outcomes

Through this project, I developed a practical understanding of:

* Low-rank matrix decomposition for neural network adaptation.
* Parameter-efficient fine-tuning.
* Transformer attention projection implementations.
* Freezing and selectively training model parameters.
* Gradient flow through custom PyTorch modules.
* Causal language modeling and validation loss.
* Experimental comparison between baseline and adapted models.
* The importance of documenting implementation differences when studying research papers.

## Limitations and Future Work

The experiment is conducted on a small dataset subset and under computational constraints. The target-module configuration and evaluation process differ from the original paper, so the results should be interpreted as an independent educational experiment rather than a direct reproduction.

Potential improvements include experimenting with different LoRA ranks, comparing target modules, training for additional epochs, introducing a held-out test set, comparing against full fine-tuning, and evaluating the model using more comprehensive benchmarks.

## Conclusion

This project provided a practical pathway from understanding the LoRA concept to implementing and integrating a custom low-rank adaptation mechanism into a pretrained language model.

It demonstrates the relationship between research concepts, model architecture, implementation details, and empirical evaluation.
