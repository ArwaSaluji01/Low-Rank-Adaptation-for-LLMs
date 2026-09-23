# Low-Rank Adaptation of DistilGPT2

A research-oriented implementation of **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning of DistilGPT2 on instruction-following data.

This project implements the core LoRA mechanism from scratch using PyTorch and integrates low-rank trainable matrices into the GPT-2-style attention projection layers of DistilGPT2. The experiment investigates how a small number of trainable parameters can adapt a pretrained language model to an instruction-formatted dataset.

## Project Overview

Full fine-tuning updates all parameters of a pretrained language model, which can be computationally expensive and memory-intensive. LoRA addresses this challenge by freezing the pretrained weights and introducing trainable low-rank matrices that represent an adaptation to the original weight matrices.

This project focuses on:

* Understanding the mathematical intuition behind LoRA.
* Implementing a LoRA layer using PyTorch.
* Integrating LoRA into DistilGPT2 attention projections.
* Fine-tuning the model on instruction-formatted data.
* Comparing baseline and LoRA-fine-tuned model performance.
* Analyzing parameter efficiency and generated responses.
* Comparing implementation choices with the original LoRA research paper.

This repository is an independent educational implementation and experimental study. It is not intended to reproduce the original paper's experiments exactly.

## Original Paper

**Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen.

The paper introduces LoRA, a parameter-efficient adaptation method that freezes pretrained model weights and trains low-rank decomposition matrices. The authors investigate LoRA across language-model adaptation tasks and analyze its relationship with low-rank structure in model updates.

The original paper discusses applying LoRA to Transformer attention projections, including query and value projection matrices. This implementation adapts the combined GPT-2-style `c_attn` projection used by DistilGPT2.

## Dataset

**Dataset:** [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

The dataset contains instruction-following examples with fields such as:

* Instruction
* Context
* Response
* Category

The project formats the examples into a causal language-modeling prompt structure:

```text
Instruction: ...
Context: ...
Response: ...
```

For examples without additional context:

```text
Instruction: ...
Response: ...
```

### Dataset Subset

| Configuration           | Value |
| ----------------------- | ----- |
| Training examples       | 2,000 |
| Validation examples     | 200   |
| Maximum sequence length | 256   |
| Random seed             | 42    |

The experiment uses a subset of the dataset because of computational and hardware constraints.

## Model and LoRA Configuration

| Configuration           | Value                   |
| ----------------------- | ----------------------- |
| Base model              | `distilgpt2`            |
| Framework               | PyTorch                 |
| Base architecture       | GPT-2-style Transformer |
| LoRA target module      | `c_attn`                |
| LoRA rank               | 4                       |
| LoRA alpha              | 4                       |
| Scaling factor          | alpha / rank            |
| Training epochs         | 1                       |
| Learning rate           | 2e-4                    |
| Batch size              | 2                       |
| Maximum sequence length | 256                     |

### Implementation Details

The custom LoRA layer retains the original pretrained projection layer and adds a low-rank update:

$$
W' = W + \frac{\alpha}{r}BA
$$

where:

* \(W\) is the frozen pretrained weight matrix.
* \(A\) and \(B\) are trainable low-rank matrices.
* \(r\) is the LoRA rank.
* \(\alpha\) controls the scaling of the update.

The implementation initializes matrix \(B\) with zeros so that the initial LoRA contribution is zero.

## Repository Contents

```text
.
├── README.md
├── summary.md
├── lora_distilgpt2.ipynb
├── lora_training_loss.png
├── finetuned_evaluation_results.txt
├── results.json
└── requirements.txt
```

> File names can be adjusted to match the actual files uploaded to the repository.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/lora-distilgpt2-from-scratch.git
cd lora-distilgpt2-from-scratch
```

### 2. Install Dependencies

The notebook is designed for execution in Google Colab.

Install the required packages:

```bash
pip install -U transformers datasets accelerate sentencepiece
```

The implementation uses:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* NumPy
* Matplotlib

### 3. Run the Notebook

Open the notebook in Google Colab or a compatible Jupyter environment.

Execute the notebook cells in order to:

1. Install dependencies.
2. Load the model and dataset.
3. Prepare the instruction-formatted data.
4. Inspect the baseline model.
5. Implement a standalone LoRA layer.
6. Integrate LoRA into DistilGPT2.
7. Verify trainable parameters and gradient flow.
8. Fine-tune the model.
9. Evaluate baseline and fine-tuned performance.
10. Visualize training progress.

## Results

The experiment evaluates the baseline DistilGPT2 model against the LoRA-fine-tuned model using validation loss, perplexity, parameter counts, and qualitative response comparisons.

### Quantitative Results

The values below will be updated after the final training and evaluation run.

| Metric                         | Baseline | LoRA Fine-Tuned |
| ------------------------------ | -------: | --------------: |
| Validation loss                |    `TBD` |           `TBD` |
| Perplexity                     |    `TBD` |           `TBD` |
| Total parameters               |    `TBD` |           `TBD` |
| Trainable parameters           |    `TBD` |           `TBD` |
| Trainable parameter percentage |    `TBD` |           `TBD` |

### Parameter Efficiency

The implementation freezes the pretrained model weights and updates only the LoRA matrices.

Final parameter statistics:

```text
Total model parameters: TBD
Trainable LoRA parameters: TBD
Trainable percentage: TBD
```

### Qualitative Evaluation

The model is evaluated using prompts related to:

* Machine learning
* Personal finance and saving
* Stocks and bonds

The comparison will examine:

* Relevance to the instruction
* Coherence
* Repetition
* Completeness
* Instruction-following behavior

Generated responses are qualitative evidence and should not be interpreted as a substitute for a comprehensive language-model evaluation.

### Training Visualization

The training-loss plot will be added to the repository after the final experiment.

```text
Training loss plot: TBD
```

## Comparison with the Original Paper

This project implements the core idea of LoRA but differs from the original paper's experimental setup.

| Aspect             | Original Paper                                   | This Implementation                                  |
| ------------------ | ------------------------------------------------ | ---------------------------------------------------- |
| Adaptation method  | Low-rank adaptation                              | Low-rank adaptation                                  |
| Base models        | Multiple Transformer models                      | DistilGPT2                                           |
| Target projections | Includes attention projections such as Wq and Wv | Combined GPT-2-style `c_attn`                        |
| Training objective | Task-specific adaptation experiments             | Instruction-formatted causal language modeling       |
| Dataset            | Paper-specific benchmark datasets                | Databricks Dolly 15k subset                          |
| Scale              | Research experiments across model sizes          | Small-scale educational experiment                   |
| Evaluation         | Task-specific benchmark metrics                  | Validation loss, perplexity, and qualitative outputs |

The comparison is methodological rather than a direct reproduction of the reported benchmark results.

The original paper investigates LoRA across several model architectures and tasks. This project focuses on understanding and implementing the underlying mechanism in a smaller experimental setting.

## What I Learned

### 1. Low-Rank Parameterization

I learned how a large weight update can be represented using two smaller trainable matrices. This provides an intuitive understanding of how LoRA reduces the number of parameters that need to be optimized.

### 2. Parameter-Efficient Fine-Tuning

I learned how freezing pretrained weights and training only adaptation parameters can reduce the number of trainable parameters while retaining the original model architecture.

### 3. Transformer Attention Projections

I investigated the attention implementation in DistilGPT2 and identified the combined `c_attn` projection used for query, key, and value computation.

This highlighted an important difference between the abstract attention formulation commonly used in research papers and the concrete implementation of Transformer architectures.

### 4. Gradient Flow and Initialization

I implemented and tested LoRA parameter initialization, including the zero initialization of matrix B. I also verified that gradients flow to the trainable LoRA parameters.

### 5. Experimental Evaluation

I learned how to compare a baseline model with a fine-tuned model using validation loss, perplexity, parameter counts, and qualitative generated outputs.

### 6. Research Reproducibility

The project reinforced the importance of documenting datasets, model configurations, training settings, evaluation procedures, and implementation differences when attempting to understand or reproduce research papers.

## Future Improvements

* Train for additional epochs and evaluate the effect on validation loss.
* Experiment with different LoRA ranks and scaling factors.
* Compare different attention target modules.
* Investigate separate query and value projection adaptation.
* Compare LoRA with full fine-tuning under a controlled experimental setup.
* Use a larger training subset or the complete dataset where hardware permits.
* Introduce a held-out test set for final evaluation.
* Evaluate generated responses using automated and human evaluation criteria.
* Add multiple random seeds and report variation across runs.
* Compare the implementation with a standard PEFT-based LoRA implementation.
* Improve checkpointing and support resuming interrupted training.
* Investigate quantization and other parameter-efficient fine-tuning methods.

## Limitations

* The experiment uses a relatively small subset of the dataset.
* Training is conducted using a limited computational setup.
* The implementation targets the combined `c_attn` projection rather than reproducing the paper's exact target-module configuration.
* The evaluation prompts are limited and do not represent a comprehensive benchmark.
* Qualitative outputs can vary depending on sampling settings and random seeds.
* Results should not be interpreted as a direct reproduction of the original paper's reported performance.

## Citation

If you reference the original LoRA method, cite:

```bibtex
@inproceedings{
    hu2022lora,
    title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
    author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```

## Acknowledgements

* The authors of the original LoRA paper.
* Hugging Face for the Transformers and Datasets libraries.
* Databricks for the Dolly instruction-following dataset.
