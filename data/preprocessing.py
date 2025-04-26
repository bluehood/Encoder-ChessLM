import chess
import pickle
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import re

class ChessBoard:
    def __init__(self):
        self.piece_values = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Black pieces
        }
        self.board = chess.Board()

    def reset(self):
        """Reset the board to initial position"""
        self.board.reset()

    def get_matrix(self) -> np.ndarray:
        """Return the current board state as a matrix"""
        matrix = np.zeros((8, 8), dtype=int)
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                symbol = piece.symbol()
                matrix[rank, file] = self.piece_values[symbol]
        
        return matrix
    
    def set_fen(self, fen: str):
        """Set board state from FEN string (Added for fallback)."""
        try:
            self.board = chess.Board(fen)
        except ValueError as e:
                print(f"Error setting FEN in fallback ChessBoard: {fen} - {e}")
                # Optionally reset to default or raise error
                self.board = chess.Board()
                raise e # Re-raise to signal the problem

def generate_mpp_sample(board_matrix: np.ndarray, turn: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate a masked piece prediction sample"""
    # Find non-zero positions (pieces)
    pieces = np.nonzero(board_matrix)
    pieces = list(zip(pieces[0], pieces[1]))
    
    # Determine number of pieces to mask (10% with minimum of 1)
    num_pieces = len(pieces)
    num_mask = max(1, int(0.1 * num_pieces))
    
    # Select random pieces to mask
    mask_indices = np.random.choice(len(pieces), num_mask, replace=False)
    mask_positions = [pieces[i] for i in mask_indices]
    
    # Create masked board and target values
    masked_board = board_matrix.copy()
    target_values = np.zeros((num_mask, 1), dtype=int)
    
    for i, (row, col) in enumerate(mask_positions):
        target_values[i] = masked_board[row, col]
        masked_board[row, col] = 0
    
    return masked_board, target_values, turn

def process_games(pgn_content: str, samples_per_game: int = 5) -> Tuple[List, List]:
    """Process chess games and generate training examples for both tasks"""
    mpp_dataset = []
    moves_dataset = []
    board = ChessBoard()
    
    # Split games
    games = pgn_content.split('[Event')[1:]  # Skip first empty split
    
    print("Processing 100000 games ...")
    for game in tqdm(games[:100000]):
        # Extract moves section
        moves_text = game.split(']')[-1].strip()
        moves = re.sub(r'\d+\.+\s*', '', moves_text)
        moves = re.sub(r'\{[^}]*\}', '', moves)
        moves = re.sub(r'1-0|0-1|1/2-1/2|\*', '', moves)
        moves = moves.strip().split()
        
        if len(moves) < 10:  # Skip very short games
            continue
            
        # Reset board for new game
        board.reset()
        
        # Generate random positions for sampling
        num_moves = len(moves)
        sample_positions = np.random.choice(
            range(num_moves),
            size=min(samples_per_game, num_moves),
            replace=False
        )
        
        # For moves between positions task, generate pairs of positions
        pair_positions = np.random.choice(
            range(num_moves),
            size=min(samples_per_game * 2, num_moves),
            replace=False
        )
        pair_positions.sort()  # Sort to ensure chronological order
        pair_positions = list(zip(pair_positions[::2], pair_positions[1::2]))
        
        # Track current position and generate examples
        positions_history = []
        for move_idx, move in enumerate(moves):
            try:
                # Parse and apply move
                chess_move = board.board.parse_san(move)
                
                # If this position was selected for MPP task
                if move_idx in sample_positions:
                    board_state = board.get_matrix()
                    turn = int(not board.board.turn)  # 0 for white, 1 for black
                    mpp_sample = generate_mpp_sample(board_state, turn)
                    mpp_dataset.append(mpp_sample)
                
                # Store position for moves between positions task
                positions_history.append((board.get_matrix(), int(not board.board.turn)))
                
                # Apply move
                board.board.push(chess_move)
                
            except chess.InvalidMoveError:
                break
        
        # Generate moves between positions samples
        for start_idx, end_idx in pair_positions:
            if start_idx >= len(positions_history) or end_idx >= len(positions_history):
                continue
            
            start_pos, start_turn = positions_history[start_idx]
            end_pos, end_turn = positions_history[end_idx]
            num_moves = end_idx - start_idx
            
            moves_dataset.append((start_pos, start_turn, end_pos, end_turn, num_moves))
    
    return mpp_dataset, moves_dataset

def save_dataset(dataset: List[Tuple], filename: str):
    """Save dataset to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def main():
    # Read PGN file
    print("Reading PGN file...")
    with open('./games_dataset.pgn', 'r') as f:
        games_content = f.read()

    # Process games and generate datasets
    mpp_dataset, moves_dataset = process_games(games_content)
    
    # Save datasets
    save_dataset(mpp_dataset, 'mpp_dataset.pkl')
    save_dataset(moves_dataset, 'moves_dataset.pkl')
    
    print(f"Saved {len(mpp_dataset)} MPP samples")
    print(f"Saved {len(moves_dataset)} moves between positions samples")

def debug():
    import debugpy
    debugpy.listen(('127.0.0.1', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    return

if __name__ == "__main__":
    # debug()
    main()