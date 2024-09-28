# llama_3_exp

## Llama 3.2 Test Project
This repository contains experiments and test cases for the Llama 3.2 model from Meta. The Llama 3.2 models include various configurations tailored for different use cases, such as general-purpose pretrained models, instruction-following fine-tuned versions, and trust and safety variants.

## Models Available
Meta offers several variants of the Llama 3.2 model:

### Pretrained Models
Llama-3.2-1B: 1 billion parameters
Llama-3.2-3B: 3 billion parameters
These models serve as a base for further fine-tuning or domain-specific adaptations.

### Instruction-Following Models
Llama-3.2-1B-Instruct: Fine-tuned to respond accurately to specific instructions
Llama-3.2-3B-Instruct: Designed for chat applications and instruction-based interactions
Trust and Safety Models
Llama-Guard-3-1B: Specialized for trust and safety applications
Llama-Guard-3-1B-INT4: An INT4-quantized version of the trust and safety model for optimized performance
How to Download the Llama 3.2 Models
Follow these steps to download the models:

### Install the Llama CLI


```bash
pip install llama-stack
```

List the Available Models: To see the latest models, run:
```bash
llama model list
```

For older versions or additional variants, use:
```bash
llama model list --show-all
```

Download a Specific Model: Run the following command, replacing MODEL_ID with the ID of the desired model:
```bash
llama model download --source meta --model-id MODEL_ID
```

Provide Your Unique Custom URL: During the download, you will be prompted to enter your unique custom URL. Copy and paste your URL into the prompt (clicking on the link directly will not trigger the download).


The downloaded model directory will contain the following files:

| **File Name**         | **Description**                                                     | **Purpose**                                                                 |
|-----------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `checklist.chk`       | Integrity check file containing hash values                         | Ensures the integrity and correctness of all files                          |
| `consolidated.00.pth` | PyTorch weight file with model parameters                           | Contains the trained weights of the neural network                          |
| `params.json`         | Configuration file with hyperparameters                             | Defines the architecture and settings required for loading the model        |
| `tokenizer.model`     | Tokenizer file in SentencePiece or similar format                   | Handles text preprocessing and conversion to/from tokens                    |

For more details, visit the official [Llama GitHub repository](https://github.com/meta/llama).


## Code Overview: Model Weight Inspection Script
This script demonstrates how to load and inspect the internal structure of a PyTorch model checkpoint file.
In this case the Llama-3.2-3B: 3 billion parameters.
The purpose of the script is to validate, analyze, and understand the contents of a Llama 3.2 model's weight file (.pth format). It provides detailed information about each tensor in the state dictionary, including dimensions and data type, and calculates the total number of parameters in the model.

### Purpose
Load a Model Checkpoint: The script loads the Llama 3.2-3B model weight file (consolidated.00.pth) into a state dictionary.
Inspect the Contents: Iterates through each tensor in the state dictionary, displaying key properties such as name, shape, and data type.
Convert and Analyze Tensors: Converts the weight tensors into a NumPy format to facilitate further numerical analysis.
Calculate Total Parameters: Computes the total number of parameters in the model to verify its size and correctness.
Prerequisites
Ensure you have the following library installed:

bash
Copia codice
pip install torch
How to Use the Script
Place the script in the same directory as your model file, or update the model_path variable to point to the correct file location.
Run the script using Python:
```bash
python model_inspection.py
```
### Script Walkthrough
1. Loading the Model Weights
The script attempts to load the specified model file using the torch.load method. If the file is not found or the format is incorrect, it prints an error message and exits.

```python
try:
    state_dict = torch.load(model_path, map_location='cpu')
except Exception as e:
    print(f"Error in loading file: {e}")
    exit(1)
```

Purpose: Verifies the model file is accessible and can be successfully loaded into memory.
2. Inspecting the State Dictionary
If the file is successfully loaded, the script checks if the contents are in a dictionary format (state_dict). It then iterates through each element, printing details such as:

Key (Element Name): The name of the parameter (e.g., layer1.weight).
Shape: Dimensions of the tensor (e.g., [3072, 1024]).
Type Conversion: Converts the tensor to float32 and NumPy format for additional analysis.

Example Output:
```bash
Element: layer1.weight
Values:
tensor([[ 0.321, -0.452,  0.892, ...], [ 0.132,  0.421, -0.581, ...]])
Value (float32):
[[ 0.321, -0.452,  0.892, ...], [ 0.132,  0.421, -0.581, ...]]
Dim weights: torch.Size([3072, 1024])
Dim weights (float32): (3072, 1024)
Dim weights (NumPy): (3072, 1024)
```

3. Calculating Total Parameters
The script sums up the total number of parameters across all tensors:

```python
total_params = 0
for key, value in state_dict.items():
    total_params += value.numel()
```

4. Output: The script prints the total number of parameters, which is crucial for validating the model's size:
```python
Total Parameters: 2,814,345,216
```

5. Expected Results
Successful Loading: If the model file is correctly loaded, you will see the details of each parameter, including names, shapes, and type conversions.
Total Parameter Count: The script outputs the total number of parameters, which helps verify if the model matches the expected architecture.

6. Sample Output
Below is a sample output from running the script on a Llama 3.2-3B model checkpoint:

```bash
Elements in dict:

Element: transformer.wte.weight
Values:
tensor([[ 0.1562, -0.3451, ...], ...])
Value (float32):
[[ 0.1562, -0.3451, ...], ...]

Dim weights: torch.Size([128256, 3072])
Dim weights (float32): (128256, 3072)
Dim weights (NumPy): (128256, 3072)

Parameter: transformer.wte.weight, Curr: 0   Dimension: torch.Size([128256, 3072])
Parameter: transformer.wpe.weight, Curr: 1   Dimension: torch.Size([1024, 3072])
...

Total Parameters: 2,814,345,216
```

## Llama 3.2 Model Architecture Overview
The Llama 3.2 model consists of a multi-layer Transformer architecture. Each layer contains multiple components, such as attention mechanisms and feed-forward networks, contributing to the model's overall complexity. Below is a breakdown of the various parameter groups and their dimensions:

### Model Structure
The Llama 3.2 model is composed of 28 Transformer layers, each with attention and feed-forward sublayers. The model also includes shared token embeddings and an output layer, as well as normalization layers applied at different stages.

### Key Components by Layer
Token Embeddings

#### tok_embeddings.weight: [128256, 3072]
Purpose: Maps the input vocabulary tokens into 3072-dimensional embeddings.
Attention Mechanism (Per Layer)

#### Weight Matrices for Self-Attention:
attention.wq.weight: [3072, 3072]

attention.wk.weight: [1024, 3072]

attention.wv.weight: [1024, 3072]

attention.wo.weight: [3072, 3072]

Purpose: Each attention sublayer uses these matrices to compute query, key, and value representations for the tokens, enabling the model to focus on different parts of the input sequence.

#### Feed-Forward Network (FFN) (Per Layer)
feed_forward.w1.weight: [8192, 3072]

feed_forward.w3.weight: [8192, 3072]

feed_forward.w2.weight: [3072, 8192]

Purpose: Performs a series of linear transformations and non-linear activations to project the hidden states to higher dimensions, enhancing the model’s representational capacity.

#### Layer Normalization (Per Layer)
attention_norm.weight: [3072]

ffn_norm.weight: [3072]

Purpose: Normalizes the output of the attention and FFN sublayers, helping to stabilize training and improve performance.

#### Output Layer
output.weight: [128256, 3072]
Purpose: Maps the final hidden states back to the vocabulary space for language modeling tasks.

### Summary
The Llama 3.2 model is highly modular, with a repeating structure of multi-head self-attention, feed-forward networks, and normalization layers. Understanding these components helps developers optimize and fine-tune the model for specific NLP tasks.

## Filename: params.json
This file defines the configuration parameters for the Llama 3.2 model. Below is a detailed explanation of each parameter and its purpose:

"dim": 3072
This parameter sets the dimension of the hidden states in the model. It determines the size of the vectors used to represent each token in the input sequence. A higher dimension typically means the model can capture more intricate features, but it also increases memory usage and computational cost.

"ffn_dim_multiplier": 1.0
Defines the multiplier for the Feed-Forward Network (FFN) dimension within each Transformer layer. The FFN dimension is calculated as 2 * dim * ffn_dim_multiplier. In this case, since the multiplier is set to 1.0, the FFN dimension is 2 * 3072 = 6144.

"multiple_of": 256
Indicates that certain dimensions (such as the FFN size) should be multiples of 256. This ensures compatibility and optimization on modern GPU architectures, which often perform better with tensor dimensions aligned to specific multiples.

"n_heads": 24
Specifies the number of attention heads in each multi-head attention layer. Each head can focus on different parts of the input sequence simultaneously, making the model capable of capturing diverse aspects of context.

"n_kv_heads": 8
Defines the number of key-value heads in the attention mechanism. Unlike the query heads (n_heads), the number of key-value heads is often smaller to optimize memory usage while preserving the ability to capture complex dependencies.

"n_layers": 28
Sets the total number of Transformer layers (or blocks) in the model. Each layer includes a self-attention sublayer followed by a feed-forward sublayer, stacked to increase the model’s depth and ability to learn complex patterns.

"norm_eps": 1e-05
A small constant added to the denominator in layer normalization operations to avoid division by zero and improve numerical stability. This helps stabilize the training process and prevents the model from generating extreme outputs.

"rope_theta": 500000.0
Determines the scaling factor for Rotary Position Embeddings (RoPE), a method used to encode positional information into token embeddings. This value controls how the model handles long sequences and positional dependencies.

"use_scaled_rope": true
A boolean flag indicating whether to use a scaled version of RoPE embeddings. Using a scaled variant can enhance the model’s performance on tasks that require understanding long-range dependencies.

"vocab_size": 128256
Specifies the size of the vocabulary the model can handle. This means the model can recognize and process 128,256 unique tokens, including words, subwords, and special symbols.

Purpose of params.json
The params.json file is essential for correctly constructing and initializing the Llama 3.2 model. It ensures that all components (e.g., attention heads, feed-forward dimensions) are aligned and configured as intended. This configuration file is used during the model’s initialization to set up the network architecture before loading the trained weights.

## Note on the Tokenizer
The tokenizer is a critical component of the model used to convert raw text into a numerical format (tokens) that the model can process. It maps characters, words, or subword units to specific numerical indices, as defined in its internal vocabulary file. In the example provided, the structure of the vocabulary is shown with a mapping between subword units (e.g., "IQ==") and their corresponding numerical IDs.

Tokenizer Vocabulary Mapping
The format displayed is a simple key-value pair where:

The key is a subword unit represented as a string, such as "IQ==" or "JA==".
The value is an integer ID starting from 0 and incrementing sequentially (e.g., 0, 1, 2, etc.).
Example Entries:
"IQ==" -> 0: This means the subword "IQ==" is mapped to the index 0.
"Jg==" -> 5: The subword "Jg==" corresponds to the index 5.
"NA==" -> 19: The subword "NA==" is assigned the index 19.
Purpose of This Structure
This type of mapping serves as the tokenizer's lookup table. When the tokenizer processes text, it segments the input into subwords (based on this vocabulary) and replaces each subword with its corresponding index. This allows the text to be represented as a sequence of integers that the model can then interpret.

Tokenizer Use Case
When input text, such as "This is an example.", is passed through the tokenizer:

The text is split into subwords or tokens.
Each subword is looked up in this mapping table.
The corresponding indices are returned as a numerical representation of the text, which the model can use for processing.
Importance of Consistency
For successful model inference or fine-tuning, the tokenizer used during preprocessing must match the one used during training. If there are discrepancies in the vocabulary or the token mappings, the model might generate incorrect outputs or fail to process inputs properly.




