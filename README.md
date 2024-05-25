# KeyBERT Emulator with GGUF Model

This project is a KeyBERT emulator that identifies the top 5 keywords from a given text passage. It uses cosine similarity and dot product to measure the similarity between the passage and each word within it. The embeddings are generated using the Llama model in the GGUF format.This project was undertaken as the orginal KeyBERT model didn't have the ability to use the GGUF model.Feel free to contribute !

## Features

- Computes embeddings for an entire text passage.
- Tokenizes the passage into individual words and computes embeddings for each word.
- Calculates the cosine similarity and dot product between the passage embedding and each word embedding.
- Identifies the top 5 keywords based on cosine similarity and dot product.
- Tracks the execution time of the process.

## Requirements

- Python 3.x
- `numpy`
- `scikit-learn`
- `llama_cpp`

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Nikil-D-Gr8/KeyBert-emulator-with-Bert.git
    ```

2. Install the required Python packages:

    ```bash
    pip install numpy scikit-learn llama_cpp
    ```

## Usage

1. Ensure the Llama model in the GGUF format (`bge-small-en-1.5-Q_4_K_M.gguf`) is available in your working directory.

2. Run the main.py file.


## Example Output

```
Top 5 words by cosine similarity: ['intelligence', 'technology', 'capabilities', 'including', 'artificial']
Top 5 words by dot product: ['intelligence', 'technology', 'capabilities', 'including', 'artificial']
Time taken to run the program: 2.34 seconds
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [KeyBERT](https://github.com/MaartenGr/KeyBERT) for the inspiration.
- The developers of [Llama](https://github.com/facebookresearch/llama) for the GGUF model.

---
