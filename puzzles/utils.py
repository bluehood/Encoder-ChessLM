import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import argparse
import numpy as np
import chess

class ChessEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.patch_embed = nn.Linear(1, d_model)
        self.turn_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, board_state, turn):
        batch_size = board_state.size(0)
        x = board_state.view(batch_size, 64, 1).float() # Ensure float input
        x = self.patch_embed(x)
        x = x + self.pos_embed
        turn_emb = self.turn_embed(turn).unsqueeze(1).expand(-1, 64, -1)
        x = x + turn_emb
        # The encoder output itself is the sequence embedding
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x

    def encode_position(self, board_matrix, turn_value):
        """Generates a single embedding vector for a board position and turn.
           This requires an adaptation or assumption about how to get a single vector
           from the sequence output (e.g., average pooling, CLS token if used).
           Here, we'll average the sequence output for simplicity.
        """
        self.eval() # Ensure model is in eval mode
        with torch.no_grad():
            # Prepare tensors
            board_tensor = torch.tensor(board_matrix, dtype=torch.float32).unsqueeze(0) # Add batch dim
            turn_tensor = torch.tensor([turn_value], dtype=torch.long)

            # Move tensors to the same device as the model parameters
            device = next(self.parameters()).device
            board_tensor = board_tensor.to(device)
            turn_tensor = turn_tensor.to(device)

            # Get sequence embedding from forward pass
            sequence_embedding = self.forward(board_tensor, turn_tensor)

            # Pool the sequence embedding (e.g., mean pooling)
            # Exclude positional embedding if needed, or pool across the 64 squares
            pooled_embedding = torch.mean(sequence_embedding, dim=1) # Pool across the sequence length (64)

            return pooled_embedding.squeeze(0).cpu().numpy() # Remove batch dim and move to CPU

class ChessBoardHelper:
    """Helper class to manage chess board state and conversion."""
    def __init__(self):
        self.piece_values = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        self.board = chess.Board()

    def set_fen(self, fen: str):
        """Set board state from FEN string."""
        try:
            self.board = chess.Board(fen)
        except ValueError as e:
            print(f"Error: Invalid FEN string: {fen} - {e}")
            raise

    def get_matrix(self) -> np.ndarray:
        """Get the board state as an 8x8 numpy matrix suitable for the model."""
        matrix = np.zeros((8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                symbol = piece.symbol()
                matrix[rank, file] = self.piece_values[symbol]
        # The model expects a flattened (64,) or (batch, 64, 1) input typically
        # Flattening it here to match potential input expectations
        return matrix.flatten() # Return shape (64,)

    def get_turn_value(self) -> int:
        """Returns 0 for White's turn, 1 for Black's turn."""
        return 0 if self.board.turn == chess.WHITE else 1