import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 1. Define the dataset class
class WeatherDataset(Dataset):
    def __init__(self, data, context_size=7, target_col='rainfall'):
        """
        Args:
            data: pandas DataFrame with weather data
            context_size: number of days to use as context
            target_col: name of the target column to predict
        """
        self.data = data
        self.context_size = context_size
        self.target_col = target_col
        
        # Identify feature columns (exclude target)
        self.feature_cols = [col for col in data.columns if col != target_col]
        self.num_features = len(self.feature_cols)
        
        # Calculate valid indices (need context_size previous days)
        self.valid_indices = range(context_size, len(data))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual index in the dataframe
        actual_idx = self.valid_indices[idx]
        
        # Get context window (current day and previous days)
        # Indexed as [day, day - 1, day - 2, ..., day - (context_size - 1)]
        context_indices = range(actual_idx - self.context_size + 1, actual_idx + 1)
        
        # Extract features for the context window (exclude rainfall)
        # Shape: [context_size, num_features]
        features = torch.tensor(
            [
                self.data.iloc[i][self.feature_cols].values 
                for i in context_indices
            ], 
            dtype=torch.float32
        )
        
        # Get target (rainfall for the current day)
        target = torch.tensor(
            self.data.iloc[actual_idx][self.target_col], 
            dtype=torch.float32
        )
        
        return features, target


# 2. Define the Transformer for Time Series Classification
class WeatherTransformer(nn.Module):
    def __init__(self, 
                 num_features,
                 d_model=64,            # Dimension of transformer
                 nhead=8,               # Number of attention heads
                 num_encoder_layers=6,  # Number of transformer layers
                 dim_feedforward=256,   # Feedforward dimension in transformer
                 dropout=0.1,
                 context_size=7):
        
        super(WeatherTransformer, self).__init__()
        
        # Feature embedding projects raw features to transformer dimension
        self.feat_emb = nn.Linear(num_features, d_model)
        self.pos_emb = nn.Embedding(context_size, d_model)
       
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: our data format is [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head (predicts rainfall probability for day 0)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # For binary classification (rainfall or not)
        )
    
    
    def forward(self, x):
        """
        Args:
            x: input features [batch_size, context_size, num_features]
        Returns:
            rainfall probability for day 0
        """

        feat_emb = self.feat_emb(x)                                   # [batch_size, context_size, d_model]
        pos_emb = self.pos_emb(torch.arange(x.size(1)).to(x.device))  # [batch_size, context_size, d_model]
        x = feat_emb + pos_emb                                        # [batch_size, context_size, d_model]

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)  # [batch_size, context_size, d_model]
        
        # Extract day 0 representation (last day in the sequence)
        day0_representation = transformer_output[:, -1, :]  # [batch_size, d_model]
        
        # Predict rainfall probability
        rainfall_prob = self.classifier(day0_representation)  # [batch_size, 1]
        
        return rainfall_prob.squeeze()  # [batch_size]


# 3. Training Function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Calculate accuracy (for binary classification)
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model


# 4. Example Usage
def example_usage():
    # Example data (would be your actual weather dataset)
    # Note: This is just placeholder data
    np.random.seed(42)
    dates = pd.date_range(start='1/1/2020', periods=1000)
    data = pd.DataFrame({
        'date': dates,
        'day_of_year': dates.dayofyear,
        'air_pressure': np.random.normal(1013, 10, 1000),
        'max_temp': np.random.normal(25, 8, 1000),
        'min_temp': np.random.normal(15, 5, 1000),
        'rainfall': (np.random.random(1000) > 0.7).astype(float)  # Binary for simplicity
    })
    
    # Create dataset and dataloaders
    context_size = 7  # Use 7 days of history
    dataset = WeatherDataset(data, context_size=context_size)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    num_features = len(dataset.feature_cols)
    model = WeatherTransformer(
        num_features=num_features,
        context_size=context_size
    )
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Save model
    torch.save(trained_model.state_dict(), 'weather_transformer.pth')
    
    # Inference example
    model.eval()
    with torch.no_grad():
        # Example: Get one sample from validation set
        sample_features, sample_target = next(iter(val_loader))
        sample_features = sample_features[0].unsqueeze(0)  # Add batch dimension
        
        # Predict
        prediction = model(sample_features)
        print(f"Predicted rainfall probability: {prediction.item():.4f}")
        print(f"Actual rainfall: {sample_target[0].item()}")


if __name__ == "__main__":
    example_usage()