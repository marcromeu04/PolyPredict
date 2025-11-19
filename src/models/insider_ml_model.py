"""
Machine Learning Model for Insider Trading Detection
Uses LSTM and Random Forest for advanced pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TradingSequenceDataset(Dataset):
    """PyTorch Dataset for trading sequences"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMInsiderDetector(nn.Module):
    """
    LSTM Neural Network for detecting insider trading patterns
    in time series trading data
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(LSTMInsiderDetector, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        out = self.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc3(out))

        return out


class InsiderMLModel:
    """
    Combined ML model using LSTM and Random Forest
    for insider trading detection
    """

    def __init__(
        self,
        sequence_length: int = 50,
        lstm_hidden_size: int = 128,
        model_dir: str = "data/models"
    ):
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()

        # Training history
        self.training_history = {
            'lstm_loss': [],
            'rf_accuracy': []
        }

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from trades data

        Features:
        - Trade frequency
        - Volume patterns
        - Price impact
        - Time-based features
        - Trader behavior metrics
        """
        if trades_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame()

        # Ensure timestamp is datetime
        if 'timestamp' in trades_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

            # Time-based features
            features['hour'] = trades_df['timestamp'].dt.hour
            features['day_of_week'] = trades_df['timestamp'].dt.dayofweek
            features['is_weekend'] = trades_df['timestamp'].dt.dayofweek >= 5

        # Volume features
        if 'size' in trades_df.columns:
            features['trade_size'] = trades_df['size']
            features['log_size'] = np.log1p(trades_df['size'])

            # Rolling statistics
            features['size_rolling_mean'] = trades_df['size'].rolling(
                window=10, min_periods=1
            ).mean()
            features['size_rolling_std'] = trades_df['size'].rolling(
                window=10, min_periods=1
            ).std()

        # Price features
        if 'price' in trades_df.columns:
            features['price'] = trades_df['price']
            features['price_change'] = trades_df['price'].pct_change()
            features['price_volatility'] = trades_df['price'].rolling(
                window=10, min_periods=1
            ).std()

        # Fill NaN values
        features = features.fillna(0)

        return features

    def create_sequences(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM training

        Args:
            features: Feature array (n_samples, n_features)
            labels: Optional labels for supervised learning

        Returns:
            Tuple of (sequences, sequence_labels)
        """
        sequences = []
        sequence_labels = []

        for i in range(len(features) - self.sequence_length):
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)

            if labels is not None:
                # Label is whether insider trading occurred in next window
                sequence_labels.append(labels[i + self.sequence_length])

        sequences = np.array(sequences)

        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        else:
            return sequences, None

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train LSTM model

        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        input_size = X_train.shape[2]

        # Initialize model
        self.lstm_model = LSTMInsiderDetector(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)

        # Create datasets
        train_dataset = TradingSequenceDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Training loop
        self.lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                # Forward pass
                outputs = self.lstm_model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.training_history['lstm_loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        logger.info("LSTM training completed")

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100
    ):
        """
        Train Random Forest classifier

        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        # Reshape if needed (flatten sequences)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)

        # Train
        self.rf_model.fit(X_train_scaled, y_train)

        # Calculate accuracy
        accuracy = self.rf_model.score(X_train_scaled, y_train)
        self.training_history['rf_accuracy'].append(accuracy)

        logger.info(f"Random Forest training completed. Accuracy: {accuracy:.4f}")

    def predict(
        self,
        features: np.ndarray,
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Predict insider trading probability

        Args:
            features: Feature array
            use_ensemble: If True, combine LSTM and RF predictions

        Returns:
            Prediction probabilities
        """
        predictions = []

        # LSTM prediction
        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                lstm_pred = self.lstm_model(features_tensor).cpu().numpy()
                predictions.append(lstm_pred)

        # Random Forest prediction
        if self.rf_model is not None:
            if len(features.shape) == 3:
                features_flat = features.reshape(features.shape[0], -1)
            else:
                features_flat = features

            features_scaled = self.scaler.transform(features_flat)
            rf_pred = self.rf_model.predict_proba(features_scaled)[:, 1]
            predictions.append(rf_pred.reshape(-1, 1))

        if not predictions:
            raise ValueError("No trained models available")

        if use_ensemble and len(predictions) > 1:
            # Ensemble: average predictions
            final_pred = np.mean(predictions, axis=0)
        else:
            final_pred = predictions[0]

        return final_pred.flatten()

    def save_models(self):
        """Save trained models to disk"""
        if self.lstm_model is not None:
            torch.save(
                self.lstm_model.state_dict(),
                self.model_dir / 'lstm_model.pt'
            )
            logger.info(f"LSTM model saved to {self.model_dir / 'lstm_model.pt'}")

        if self.rf_model is not None:
            with open(self.model_dir / 'rf_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
            logger.info(f"Random Forest model saved to {self.model_dir / 'rf_model.pkl'}")

        # Save scaler
        with open(self.model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_models(self, input_size: int):
        """Load trained models from disk"""
        lstm_path = self.model_dir / 'lstm_model.pt'
        rf_path = self.model_dir / 'rf_model.pkl'
        scaler_path = self.model_dir / 'scaler.pkl'

        if lstm_path.exists():
            self.lstm_model = LSTMInsiderDetector(
                input_size=input_size,
                hidden_size=self.lstm_hidden_size
            ).to(self.device)
            self.lstm_model.load_state_dict(torch.load(lstm_path))
            self.lstm_model.eval()
            logger.info("LSTM model loaded")

        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            logger.info("Random Forest model loaded")

        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded")


if __name__ == "__main__":
    # Test the model
    model = InsiderMLModel()

    # Generate synthetic data for testing
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Create sequences
    sequences, labels = model.create_sequences(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    # Train models
    print("Training LSTM...")
    model.train_lstm(X_train, y_train, epochs=10)

    print("Training Random Forest...")
    model.train_random_forest(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    # Save models
    model.save_models()
    print("Models saved successfully")
