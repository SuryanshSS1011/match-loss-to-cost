"""
LSTM model training and forecasting for network traffic prediction.

Trains a multi-variate LSTM that jointly predicts all link loads.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, DATA_DIR, MODELS_DIR, RESULTS_DIR
from utils import set_all_seeds, make_sequences, compute_metrics, aggregate_metrics, save_json


class LSTMForecaster(nn.Module):
    """LSTM model for multi-variate time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """
        Args:
            input_size: Number of input features (num_links)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, window_size, num_links)

        Returns:
            Output tensor of shape (batch, num_links)
        """
        # LSTM forward pass
        out, (h_n, c_n) = self.lstm(x)

        # Use last time step output
        out = self.fc(out[:, -1, :])

        return out


def load_and_prepare_data(window_size: int):
    """
    Load data and prepare sequences for LSTM training.

    Returns:
        train_loader, val_loader, test data, normalization stats
    """
    # Load data
    data = np.load(os.path.join(DATA_DIR, 'traffic_data.npz'))
    L = data['L']
    train_end = int(data['train_end'])
    val_end = int(data['val_end'])

    # Split data
    L_train = L[:train_end]
    L_val = L[train_end:val_end]
    L_test = L[val_end:]

    # Compute normalization stats from training data only
    mean = L_train.mean(axis=0)
    std = L_train.std(axis=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero

    # Normalize all data
    L_train_norm = (L_train - mean) / std
    L_val_norm = (L_val - mean) / std
    L_test_norm = (L_test - mean) / std

    # Create sequences
    X_train, y_train = make_sequences(L_train_norm, window_size)
    X_val, y_val = make_sequences(L_val_norm, window_size)
    X_test, y_test = make_sequences(L_test_norm, window_size)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    batch_size = CONFIG['lstm_batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get corresponding true test values (denormalized)
    # y_test corresponds to L_test[window_size:]
    L_test_aligned = L_test[window_size:]

    return (train_loader, val_loader, X_test_t, y_test_t,
            L_test_aligned, mean, std)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model(model, train_loader, val_loader, device, config):
    """
    Train LSTM model with early stopping.

    Returns:
        Trained model, training history
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lstm_lr'])

    epochs = config['lstm_epochs']
    patience = config['lstm_patience']

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n   Training for up to {epochs} epochs (patience={patience})...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    print(f"   Best validation loss: {best_val_loss:.6f}")

    return model, history


def predict(model, X_test, mean, std, device):
    """
    Generate predictions on test data.

    Args:
        model: Trained LSTM model
        X_test: Test input tensor
        mean: Training mean for denormalization
        std: Training std for denormalization
        device: PyTorch device

    Returns:
        Denormalized predictions (numpy array)
    """
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        predictions_norm = model(X_test).cpu().numpy()

    # Denormalize
    predictions = predictions_norm * std + mean

    return predictions


def main():
    """Train LSTM model and generate forecasts."""
    print("=" * 50)
    print("Training LSTM Model")
    print("=" * 50)

    # Set random seeds
    set_all_seeds(CONFIG['random_seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Using device: {device}")

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    window_size = CONFIG['window_size']
    (train_loader, val_loader, X_test, y_test,
     L_test_aligned, mean, std) = load_and_prepare_data(window_size)

    num_links = mean.shape[0]
    print(f"   - Window size: {window_size}")
    print(f"   - Number of links: {num_links}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    print(f"   - Test samples: {len(X_test)}")

    # Create model
    print("\n2. Creating LSTM model...")
    model = LSTMForecaster(
        input_size=num_links,
        hidden_size=CONFIG['lstm_hidden_size'],
        num_layers=CONFIG['lstm_num_layers']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Hidden size: {CONFIG['lstm_hidden_size']}")
    print(f"   - Num layers: {CONFIG['lstm_num_layers']}")
    print(f"   - Total parameters: {total_params:,}")

    # Train model
    print("\n3. Training model...")
    model, history = train_model(model, train_loader, val_loader, device, CONFIG)

    # Save model
    print("\n4. Saving model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'lstm_forecaster.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'mean': mean,
        'std': std,
        'history': history
    }, model_path)
    print(f"   Saved to: {model_path}")

    # Save normalization stats for reproducibility and testing
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = np.load(os.path.join(DATA_DIR, 'traffic_data.npz'))
    train_end = int(data['train_end'])
    save_json({
        'mean': mean.tolist(),
        'std': std.tolist(),
        'train_end': train_end
    }, os.path.join(RESULTS_DIR, 'normalization_stats.json'))
    print(f"   Saved normalization stats to: {RESULTS_DIR}/normalization_stats.json")

    # Generate predictions
    print("\n5. Generating predictions...")
    predictions = predict(model, X_test, mean, std, device)
    print(f"   Predictions shape: {predictions.shape}")

    # Save predictions
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, 'lstm_predictions.npz'),
        predictions=predictions,
        L_test_aligned=L_test_aligned
    )

    # Compute metrics
    print("\n6. Computing metrics...")
    per_link_metrics = compute_metrics(L_test_aligned, predictions)
    aggregated = aggregate_metrics(per_link_metrics)

    # Print summary
    print("\n   LSTM Forecasting Metrics:")
    print(f"   - Mean RMSE: {aggregated['rmse_mean']:.4f}")
    print(f"   - Mean MAE:  {aggregated['mae_mean']:.4f}")
    print(f"   - Mean MAPE: {aggregated['mape_mean']:.2f}%")

    # Save metrics
    metrics_data = {
        'per_link': {
            'rmse': per_link_metrics['rmse'].tolist(),
            'mae': per_link_metrics['mae'].tolist(),
            'mape': per_link_metrics['mape'].tolist()
        },
        'aggregated': aggregated
    }
    save_json(metrics_data, os.path.join(RESULTS_DIR, 'lstm_metrics.json'))

    print("\n" + "=" * 50)
    print("LSTM training complete!")
    print("=" * 50)

    return predictions, L_test_aligned, per_link_metrics, aggregated


if __name__ == '__main__':
    main()
