# ChessLM: Contextual Chess Position Embeddings

## Project Overview

ChessLM is a Transformer-based model designed to learn rich, contextual vector representations (embeddings) for chess positions. Inspired by self-supervised learning in NLP (like BERT) and adapting the Vision Transformer (ViT) architecture, ChessLM focuses on capturing the strategic and thematic similarities between board states, rather than primarily predicting the best move or evaluating the position's score like traditional chess engines.

The core of the model is a Transformer encoder that processes the 8x8 board, considering piece types, locations (via positional embeddings), and whose turn it is (via a turn embedding). It outputs a 256-dimensional embedding vector for a given position (represented by a FEN string).

More details can be found in the technical writeup https://bluehood.github.io/research/benh_Beyond_Evaluation__Learning_Contextual_Chess_Position_Representations_2025.pdf.

## Model Architecture and Training

The model adopts an encoder Transformer architecture with 6 layers each with 8 heads. The model has approximately 4.5 million total parameters, all of which are trainable.

To encourage the model to learn comprehensive representations of chess positions, a multi-task learning strategy combining two self-supervised objectives was employed:

1.  **Masked Piece Prediction (MPP):** Analogous to BERT's Masked Language Model task, a random subset (10%) of pieces on the input board are masked. The model predicts the original identity of these masked pieces based on the surrounding context. This task helps the model understand typical piece configurations, legal placements, and piece relationships.
2.  **Moves Difference Prediction:** The model is presented with two distinct board states (start and end positions) from actual game sequences and must predict the number of moves (plies) separating them. This encourages the model to learn about piece mobility, game dynamics, and position evolution.

Training utilised datasets derived from the Lichess database and the CCRL database, pre-processed for these self-supervised tasks.

You can download the model from Huggingface at https://huggingface.co/odestorm1/chesslm. 

## Code Structure

*   **`train/`**: Contains the code for training the `ChessVisionTransformer` model (`train/train.py`) on the MPP and Moves Difference tasks. This includes the model definition, dataset classes, and the training loop.
*   **`examples/`**: Provides scripts demonstrating how to load the pre-trained encoder and generate embeddings for chess positions (FEN strings).
*   **`puzzles/`**: Includes a Flask application (`puzzles/app.py`) that uses the generated embeddings to find chess puzzles strategically or thematically similar to a given input puzzle. This uses a precompiled dataset of chess puzzles and their embeddings from Huggingface (odestorm1/chesslm_puzzles) although you can create you own using the `generate_puzzle_embeddings.py` script.
*   **`data/`**: Intended location for storing raw and processed datasets (e.g., `mpp_dataset.pkl`, `moves_dataset.pkl`). You can process your own datasets from these datasets using `preprocessing.py`.

## Intended Uses & Limitations

### Intended Use

The primary intended use of this model is to generate embeddings that capture the "feel" or thematic essence of a chess position. These embeddings can be used for:

*   **Position Similarity Search:** Finding positions in a database that are structurally or strategically similar to a query position (as demonstrated in the `puzzles/` app).
*   **Retrieval-Augmented Generation (RAG):** Enhancing chess analysis tools by retrieving similar historical positions.
*   **Downstream Task Input:** Serving as input features for tasks like classifying tactical motifs or suggesting relevant puzzles.

### Limitations

*   **Not an Evaluation Engine:** ChessLM does not predict position evaluation (e.g., centipawn score). Embeddings capture structural similarities, but positions deemed similar can have vastly different engine evaluations.
*   **Focus on Structure:** The model may overemphasise structural similarities (like pawn formations) while potentially under-weighting critical dynamic factors.

## Citation

If you use this model, its embeddings, or the concepts presented, please cite:

```
@misc{hull2025beyond,
      title={Beyond Evaluation: Learning Contextual Chess Position Representations},
      author={Ben Hull},
      year={2025},
      howpublished={Accessed via \url{[https://bluehood.github.io/](https://bluehood.github.io/)}},
      note={Technical report}
}
```
