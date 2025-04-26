import os
import sys
import chess
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import random
import time
from tqdm import tqdm 
from huggingface_hub import hf_hub_download

script_dir = Path(__file__).parent
v4_dir = script_dir.parent
sys.path.append(str(v4_dir))

try:
    from utils import ChessBoardHelper, ChessEncoder
except ImportError:
    print("Error: Could not import ChessBoardHelper or ChessEncoder from utils.py.")
    print("Ensure utils.py is in the same directory or sys.path is configured correctly.")
    sys.exit(1)


# --- Configuration ---
PUZZLE_CSV_PATH = v4_dir / 'data' / 'lichess_db_puzzle.csv'
HF_MODEL_REPO_ID = "odestorm1/chesslm"
OUTPUT_DIR = script_dir
OUTPUT_FILENAME = OUTPUT_DIR / 'puzzle_embeddings.parquet'
SAMPLE_SIZE = 1_000
BATCH_SIZE = 64

# --- Global Variables ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_embeddings_batch(fens, encoder, helper, device):
    """Calculates embeddings for a batch of FEN strings."""
    batch_matrices = []
    batch_turns = []
    valid_indices = [] # Track indices of FENs that were successfully parsed

    for i, fen in enumerate(fens):
        try:
            # Use helper methods consistent with utils.py version
            board = chess.Board(fen) # Keep standard chess parsing
            matrix = helper.get_matrix(board) # Assuming get_matrix still takes a board object
            turn = helper.get_turn_int(board) # Assuming get_turn_int still takes a board object
            batch_matrices.append(matrix)
            batch_turns.append(turn)
            valid_indices.append(i)
        except ValueError:
            print(f"Warning: Skipping invalid FEN: {fen}")
            continue # Skip invalid FENs

    if not batch_matrices:
        return [], [] # Return empty if no valid FENs in batch

    board_tensor = torch.tensor(np.array(batch_matrices), dtype=torch.float32).to(device)
    turn_tensor = torch.tensor(batch_turns, dtype=torch.long).to(device)

    with torch.no_grad():
        # Use the encode method from the Hugging Face integrated ChessEncoder
        # Assuming the method is named encode_position based on app.py
        embeddings = encoder.encode_position(board_tensor, turn_tensor)

    # Detach, move to CPU, convert to numpy/list
    # Check if the output is already a tensor or needs conversion
    if isinstance(embeddings, torch.Tensor):
        embeddings_list = embeddings.cpu().numpy().tolist()
    else: # Adapt if the output format is different (e.g., BaseModelOutput)
        # Assuming output might be in a structure, adjust as needed
        # Example: embeddings_list = embeddings.last_hidden_state[:, 0].cpu().numpy().tolist()
        print("Warning: Output format from encoder.encode_position may have changed. Adjusting extraction.")
        # Fallback/Guess: assume it's the direct tensor output for now
        embeddings_list = embeddings.cpu().numpy().tolist()


    return embeddings_list, valid_indices

def main():
    # global encoder_model # No longer global
    print(f"Using device: {device}")
    print(f"Output will be saved to: {OUTPUT_FILENAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create output directory

    # --- Load Encoder Model from Hugging Face ---
    try:
        print(f"Loading encoder model from Hugging Face Hub: {HF_MODEL_REPO_ID}...")
        start_time = time.time()
        # Load using from_pretrained
        encoder_model = ChessEncoder.from_pretrained(HF_MODEL_REPO_ID)
        encoder_model.to(device) # Move model to the appropriate device
        encoder_model.eval()    # Set model to evaluation mode
        print(f"Encoder loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL ERROR loading encoder model from Hugging Face Hub: {e}")
        sys.exit(1)

    # --- Load Puzzle Data ---
    try:
        print(f"Loading puzzle dataset from: {PUZZLE_CSV_PATH}...")
        start_time = time.time()
        # Use specific columns to save memory if the CSV is large
        # Adjust columns if needed based on the actual CSV header
        use_cols = ['FEN', 'Moves', 'Themes']
        df = pd.read_csv(PUZZLE_CSV_PATH, usecols=use_cols)
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Puzzle dataset not found at {PUZZLE_CSV_PATH}")
        sys.exit(1)
    except ValueError as e:
         print(f"FATAL ERROR: Could not find required columns in CSV. Check header. Error: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR loading dataset: {e}")
        sys.exit(1)


    # --- Sample Data ---
    num_rows = len(df)
    actual_sample_size = min(SAMPLE_SIZE, num_rows)
    print(f"Sampling {actual_sample_size} puzzles from {num_rows} total puzzles...")
    if actual_sample_size < num_rows:
        df_sampled = df.sample(n=actual_sample_size, random_state=42).copy() # Use random_state for reproducibility
    else:
        df_sampled = df.copy() # Use all rows if dataset is smaller than sample size
    del df # Free memory
    print("Sampling complete.")


    # --- Calculate Embeddings ---
    print(f"Calculating embeddings for {actual_sample_size} puzzles (Batch size: {BATCH_SIZE})...")
    start_time = time.time()
    board_helper = ChessBoardHelper() # Instantiate helper from utils
    all_embeddings = []
    processed_indices = [] # Store original indices of successfully processed rows

    # Use tqdm for progress bar
    for i in tqdm(range(0, len(df_sampled), BATCH_SIZE), desc="Calculating Embeddings"):
        batch_df = df_sampled.iloc[i : i + BATCH_SIZE]
        batch_fens = batch_df['FEN'].tolist()

        # Pass the loaded encoder_model to the batch function
        embeddings_list, valid_batch_indices = calculate_embeddings_batch(
            batch_fens, encoder_model, board_helper, device
        )

        # Map valid batch indices back to original df_sampled indices
        original_indices = [batch_df.index[j] for j in valid_batch_indices]
        processed_indices.extend(original_indices)
        all_embeddings.extend(embeddings_list)


    print(f"Embeddings calculation finished in {time.time() - start_time:.2f} seconds.")
    print(f"Successfully processed {len(all_embeddings)} puzzles out of {actual_sample_size} sampled.")

    # --- Create Final DataFrame ---
    if not all_embeddings:
         print("No embeddings were generated. Exiting.")
         sys.exit(0)

    # Filter the sampled DataFrame to only include rows that were successfully processed
    df_processed = df_sampled.loc[processed_indices].copy()
    df_processed['Embedding'] = all_embeddings

    # Select and reorder columns for the final output
    output_df = df_processed[['FEN', 'Moves', 'Themes', 'Embedding']]


    # --- Save Results ---
    try:
        print(f"Saving results to {OUTPUT_FILENAME}...")
        start_time = time.time()
        output_df.to_parquet(OUTPUT_FILENAME, index=False)
        print(f"Results saved successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL ERROR saving results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 