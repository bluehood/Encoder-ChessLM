import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pickle
from tqdm import tqdm

class MPPDataset(Dataset):
    """Dataset for Masked Piece Prediction task"""
    def __init__(self, mpp_data):
        self.samples = []
        
        def map_piece_to_index(piece_value):
            return piece_value
        
        for masked_board, target_values, turn in mpp_data:
            # Pad target_values to max possible length (6) for consistent batch size
            padded_targets = np.zeros((6, 1), dtype=int)
            # Map each target value to its corresponding index
            mapped_values = np.array([map_piece_to_index(v[0]) for v in target_values])
            padded_targets[:len(mapped_values), 0] = mapped_values
            
            self.samples.append({
                'masked_board': torch.tensor(masked_board, dtype=torch.float32),
                'target_values': torch.tensor(padded_targets, dtype=torch.long),
                'target_mask': torch.tensor([1] * len(target_values) + [0] * (6 - len(target_values)), dtype=torch.bool),
                'turn': torch.tensor(turn, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class MovesDataset(Dataset):
    """Dataset for Moves Prediction task"""
    def __init__(self, moves_data):
        self.samples = []
        for start_pos, start_turn, end_pos, end_turn, num_moves in moves_data:
            self.samples.append({
                'start_pos': torch.tensor(start_pos, dtype=torch.float32),
                'start_turn': torch.tensor(start_turn, dtype=torch.long),
                'end_pos': torch.tensor(end_pos, dtype=torch.float32),
                'end_turn': torch.tensor(end_turn, dtype=torch.long),
                'num_moves': torch.tensor(num_moves, dtype=torch.float32)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class ChessVisionTransformer(nn.Module):
    """Vision Transformer for chess board processing"""
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Core encoder components
        self.patch_embed = nn.Linear(1, d_model)
        self.turn_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        # Task-specific tokens
        self.mpp_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.moves_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Task-specific heads
        self.mpp_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 12)  # -6 to -1, 1 to 6
        )
        
        self.moves_head = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_board(self, board_state, turn, task_token):
        """Encode a chess board into transformer space"""
        batch_size = board_state.size(0)
        
        # Process board state
        x = board_state.view(batch_size, 64, 1)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Add turn information
        turn_emb = self.turn_embed(turn).unsqueeze(1).expand(-1, 64, -1)
        x = x + turn_emb
        
        # Add task token
        task_tokens = task_token.expand(batch_size, -1, -1)
        x = torch.cat((task_tokens, x), dim=1)
        
        return x

def train_model(model, mpp_train_loader, mpp_val_loader, moves_train_loader, moves_val_loader, 
             num_epochs, device, weight_decay=0.01):
    """Train model on both MPP and Moves tasks simultaneously"""
    model = model.to(device)
    
    # Loss functions
    mpp_criterion = nn.CrossEntropyLoss(reduction='none')
    moves_criterion = nn.MSELoss()
    
    # Single optimizer for both tasks
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        mpp_train_loss = 0
        mpp_train_correct = 0
        mpp_train_total = 0
        moves_train_loss = 0
        moves_train_mae = 0
        
        # MPP Training Phase
        train_pbar = tqdm(mpp_train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [MPP Train]')
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            x = model.encode_board(batch['masked_board'], batch['turn'], model.mpp_token)
            x = model.transformer(x)
            task_output = model.layer_norm(x[:, 0])
            piece_logits = model.mpp_head(task_output)
            
            # Handle predictions
            piece_logits = piece_logits.unsqueeze(1).expand(-1, 6, -1)
            target_values = batch['target_values'].squeeze(-1)
            
            # Masked loss
            loss = mpp_criterion(piece_logits.reshape(-1, 12), target_values.reshape(-1))
            loss = loss.reshape(target_values.shape)
            loss = (loss * batch['target_mask']).sum() / batch['target_mask'].sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            mpp_train_loss += loss.item()
            pred = piece_logits.argmax(dim=-1)
            correct = (pred == target_values) & batch['target_mask']
            mpp_train_correct += correct.sum().item()
            mpp_train_total += batch['target_mask'].sum().item()
            
            train_pbar.set_postfix({
                'mpp_loss': loss.item(),
                'mpp_acc': 100. * mpp_train_correct / mpp_train_total
            })
        
        # Moves Training Phase
        train_pbar = tqdm(moves_train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Moves Train]')
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Process both positions
            start_x = model.encode_board(batch['start_pos'], batch['start_turn'], model.moves_token)
            end_x = model.encode_board(batch['end_pos'], batch['end_turn'], model.moves_token)
            
            start_x = model.transformer(start_x)
            end_x = model.transformer(end_x)
            
            start_output = model.layer_norm(start_x[:, 0])
            end_output = model.layer_norm(end_x[:, 0])
            
            # Predict moves
            combined = torch.cat((start_output, end_output), dim=1)
            moves_pred = model.moves_head(combined).squeeze(-1)
            
            loss = moves_criterion(moves_pred, batch['num_moves'])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            moves_train_loss += loss.item()
            moves_train_mae += torch.abs(moves_pred - batch['num_moves']).mean().item()
            
            train_pbar.set_postfix({
                'moves_loss': loss.item(),
                'moves_mae': moves_train_mae / (train_pbar.n + 1)
            })
        
        # Validation
        model.eval()
        mpp_val_loss = 0
        mpp_val_correct = 0
        mpp_val_total = 0
        moves_val_loss = 0
        moves_val_mae = 0
        
        with torch.no_grad():
            # MPP Validation
            val_pbar = tqdm(mpp_val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [MPP Val]')
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                x = model.encode_board(batch['masked_board'], batch['turn'], model.mpp_token)
                x = model.transformer(x)
                task_output = model.layer_norm(x[:, 0])
                piece_logits = model.mpp_head(task_output)
                
                piece_logits = piece_logits.unsqueeze(1).expand(-1, 6, -1)
                target_values = batch['target_values'].squeeze(-1)
                
                loss = mpp_criterion(piece_logits.reshape(-1, 12), target_values.reshape(-1))
                loss = loss.reshape(target_values.shape)
                loss = (loss * batch['target_mask']).sum() / batch['target_mask'].sum()
                
                mpp_val_loss += loss.item()
                pred = piece_logits.argmax(dim=-1)
                correct = (pred == target_values) & batch['target_mask']
                mpp_val_correct += correct.sum().item()
                mpp_val_total += batch['target_mask'].sum().item()
                
                val_pbar.set_postfix({
                    'mpp_loss': loss.item(),
                    'mpp_acc': 100. * mpp_val_correct / mpp_val_total
                })
            
            # Moves Validation
            val_pbar = tqdm(moves_val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Moves Val]')
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                start_x = model.encode_board(batch['start_pos'], batch['start_turn'], model.moves_token)
                end_x = model.encode_board(batch['end_pos'], batch['end_turn'], model.moves_token)
                
                start_x = model.transformer(start_x)
                end_x = model.transformer(end_x)
                
                start_output = model.layer_norm(start_x[:, 0])
                end_output = model.layer_norm(end_x[:, 0])
                
                combined = torch.cat((start_output, end_output), dim=1)
                moves_pred = model.moves_head(combined).squeeze(-1)
                
                loss = moves_criterion(moves_pred, batch['num_moves'])
                moves_val_loss += loss.item()
                moves_val_mae += torch.abs(moves_pred - batch['num_moves']).mean().item()
                
                val_pbar.set_postfix({
                    'moves_loss': loss.item(),
                    'moves_mae': moves_val_mae / (val_pbar.n + 1)
                })
        
        # Calculate average losses
        mpp_val_loss /= len(mpp_val_loader)
        moves_val_loss /= len(moves_val_loader)
        moves_val_mae /= len(moves_val_loader)
        
        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'MPP Train Loss: {mpp_train_loss/len(mpp_train_loader):.4f}, '
              f'Acc: {100.*mpp_train_correct/mpp_train_total:.2f}%')
        print(f'MPP Val Loss: {mpp_val_loss:.4f}, '
              f'Acc: {100.*mpp_val_correct/mpp_val_total:.2f}%')
        print(f'Moves Train Loss: {moves_train_loss/len(moves_train_loader):.4f}, '
              f'MAE: {moves_train_mae/len(moves_train_loader):.2f}')
        print(f'Moves Val Loss: {moves_val_loss:.4f}, MAE: {moves_val_mae:.2f}')
        
        # Learning rate scheduling
        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mpp_val_loss': mpp_val_loss,
            'moves_val_loss': moves_val_loss,
        }, f'best_chess_model_epoch_{epoch+1}.pt')
        
        # Save encoder separately
        save_encoder(model, f'best_chess_encoder_epoch_{epoch+1}.pt')

def save_encoder(model, filename):
    """Save just the encoder part of the model"""
    encoder_state = {
        'patch_embed': model.patch_embed.state_dict(),
        'turn_embed': model.turn_embed.state_dict(),
        'pos_embed': model.pos_embed,
        'transformer': model.transformer.state_dict(),
        'layer_norm': model.layer_norm.state_dict(),
        'hyperparameters': {
            'd_model': model.patch_embed.out_features,
            'nhead': model.transformer.layers[0].self_attn.num_heads,
            'num_layers': len(model.transformer.layers),
            'dim_feedforward': model.transformer.layers[0].linear1.out_features,
            'dropout': model.transformer.layers[0].dropout.p,
        }
    }
    torch.save(encoder_state, filename)

def main():
    # Load datasets
    print("Loading datasets...")
    with open('../data/mpp_dataset.pkl', 'rb') as f:
        mpp_data = pickle.load(f)
    with open('../data/moves_dataset.pkl', 'rb') as f:
        moves_data = pickle.load(f)
    
    # Create datasets
    mpp_dataset = MPPDataset(mpp_data)
    moves_dataset = MovesDataset(moves_data)
    
    # Split into train and validation sets (90-10 split)
    mpp_train_size = int(0.9 * len(mpp_dataset))
    mpp_val_size = len(mpp_dataset) - mpp_train_size
    mpp_train_dataset, mpp_val_dataset = random_split(mpp_dataset, [mpp_train_size, mpp_val_size])
    
    moves_train_size = int(0.9 * len(moves_dataset))
    moves_val_size = len(moves_dataset) - moves_train_size
    moves_train_dataset, moves_val_dataset = random_split(moves_dataset, [moves_train_size, moves_val_size])
    
    # Create data loaders
    mpp_train_loader = DataLoader(mpp_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    mpp_val_loader = DataLoader(mpp_val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    moves_train_loader = DataLoader(moves_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    moves_val_loader = DataLoader(moves_val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = ChessVisionTransformer(
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'Using device: {device}')
    
    train_model(
        mpp_train_loader=mpp_train_loader,
        mpp_val_loader=mpp_val_loader,
        moves_train_loader=moves_train_loader,
        moves_val_loader=moves_val_loader,
        model=model,
        num_epochs=30,
        device=device
    )

if __name__ == "__main__":
    main()