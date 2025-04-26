import os
import sys
import chess
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time
import random
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download  # Import hf_hub_download

# Add parent directories to sys.path for imports
script_dir = Path(__file__).parent
v4_dir = script_dir.parent
sys.path.append(str(v4_dir))

from utils import ChessBoardHelper, ChessEncoder

# Configuration
HF_MODEL_REPO_ID = "odestorm1/chesslm"
HF_MODEL_FILENAME = "chesslm_encoder_epoch_30.pt"
HF_DATASET_REPO_ID = "odestorm1/chesslm_puzzles"
HF_DATASET_FILENAME = "puzzle_embeddings.parquet"
EMBEDDINGS_PATH = script_dir / 'puzzle_embeddings.parquet'

# Global variables
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_model = None
puzzle_df = None
board_helper = None

def load_data():
    """Load puzzle embeddings and encoder model from Hugging Face Hub."""
    global encoder_model, puzzle_df, board_helper

    # Download and load puzzle embeddings from Hugging Face Hub
    print(f"Downloading embeddings '{HF_DATASET_FILENAME}' from dataset repo '{HF_DATASET_REPO_ID}'...")
    try:
        downloaded_embeddings_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=HF_DATASET_FILENAME,
            repo_type='dataset'  # Specify repo type as dataset
        )
        print(f"Embeddings downloaded to: {downloaded_embeddings_path}")

        puzzle_df = pd.read_parquet(downloaded_embeddings_path)
        print(f"Loaded {len(puzzle_df)} puzzles with embeddings from Hugging Face Hub.")

    except Exception as e:
        print(f"Error loading embeddings from Hugging Face Hub: {e}")
        raise RuntimeError(f"Failed to load embeddings from Hugging Face Hub: {e}") from e

    # Download and load encoder model from Hugging Face Hub
    print(f"Downloading encoder model '{HF_MODEL_FILENAME}' from Hugging Face repo '{HF_MODEL_REPO_ID}'...")
    try:
        encoder_model = ChessEncoder.from_pretrained(HF_MODEL_REPO_ID)
        encoder_model.eval() # Set to evaluation mode
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model from Hugging Face Hub: {e}")
        raise RuntimeError(f"Failed to load model from Hugging Face Hub: {e}") from e

    # Initialize chess board helper
    board_helper = ChessBoardHelper()

    print("Data loading complete.")

def get_embedding_for_fen(fen):
    """Calculate the embedding for a given FEN string."""
    try:
        board_helper.set_fen(fen)
        matrix = board_helper.get_matrix()
        turn = board_helper.get_turn_value()
        
        board_tensor = torch.tensor(np.array([matrix]), dtype=torch.float32).to(device)
        turn_tensor = torch.tensor([turn], dtype=torch.long).to(device)
        
        with torch.no_grad():
            embedding = encoder_model.encode_position(board_tensor, turn_tensor)
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding for FEN {fen}: {e}")
        raise ValueError(f"Invalid FEN or error generating embedding: {e}")

def find_similar_puzzles(fen, top_n=10):
    """Find the top_n most similar puzzles based on cosine similarity."""
    try:
        # Get the embedding for the input FEN
        input_embedding = get_embedding_for_fen(fen)
        
        # Convert list of embeddings to a numpy array
        all_embeddings = np.array(puzzle_df['Embedding'].tolist())
        
        # Calculate cosine similarity
        similarities = cosine_similarity([input_embedding], all_embeddings)[0]
        
        # Get indices of top_n most similar puzzles
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Get the most similar puzzles
        similar_puzzles = puzzle_df.iloc[top_indices].copy()
        similar_puzzles['similarity'] = similarities[top_indices]
        
        return similar_puzzles
    except Exception as e:
        print(f"Error finding similar puzzles: {e}")
        raise ValueError(f"Failed to find similar puzzles: {e}")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/find_similar_puzzle', methods=['POST'])
def find_similar_puzzle():
    """API endpoint to find a similar puzzle."""
    try:
        # Get FEN from request
        data = request.json
        fen = data.get('fen')
        
        if not fen:
            return jsonify({'detail': 'FEN string is required'}), 400
        
        # Validate FEN
        try:
            chess.Board(fen)
        except ValueError:
            return jsonify({'detail': 'Invalid FEN string'}), 400
        
        # Find similar puzzles
        similar_puzzles = find_similar_puzzles(fen)
        
        # Use current Unix time as seed for random selection
        seed = int(time.time())
        random.seed(seed)
        
        # Randomly select one puzzle from the top 10 similar puzzles
        selected_puzzle = similar_puzzles.sample(n=1, random_state=seed).iloc[0]
        
        # Extract the moves sequence and first move
        moves = selected_puzzle['Moves']
        moves_list = moves.split()
        first_move = moves_list[0]
        
        # Return the selected puzzle with complete moves sequence
        return jsonify({
            'puzzle_fen': selected_puzzle['FEN'],
            'themes': selected_puzzle['Themes'],
            'first_move': first_move,
            'moves': moves_list,
            'similarity': float(selected_puzzle['similarity'])
        })
    except ValueError as e:
        return jsonify({'detail': str(e)}), 400
    except Exception as e:
        return jsonify({'detail': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000) 