import numpy as np
import torch.nn as nn
import torch
from .load_scalers import build_features, seq_len, num_features

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1 block #1
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()

        # Conv1 block #2
        self.conv2 = nn.Conv1d(
            in_channels=32, 
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.act2 = nn.GELU()

        # Output layer
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, num_features, seq_len)

        # Conv block #1
        x = self.conv1(x)
        x = self.act1(x)

        # Conv block #2
        x = self.conv2(x)
        x = self.act2(x)

        x = x.permute(0, 2, 1) # (batch, seq_len, num_features)
        x = x[:, -1, :] # get last hidden

        # Output layer
        x = self.fc_out(x)
        return x
    
class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :] 

        # Output layer
        x = self.fc_out(x)
        return x
    
class Model_3(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1D
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()

        # LSTM
        self.lstm = nn.LSTM(64, 32, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(32, num_features)

    def forward(self, x):
        # Conv1D
        x = x.permute(0, 2, 1) # (batch, num_features, seq_len)
        x = self.conv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, num_features)

        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]

        # Output layer
        x = self.fc_out(x)
        return x


    
MODELS_INFOS = [
    # Model 1
    {"paths":
     [
        "./trading_bot/ensemble_models/eq_3778_ep_25.pt",
        "./trading_bot/ensemble_models/eq_3768_ep_80.pt",
      ],
     "architecture": Model_1},

    # Model 2
    {"paths":
     [
        "./trading_bot/ensemble_models/eq_3590_ep_122.pt",
      ],
     "architecture": Model_2},

    # Model 3
    {"paths":
     [
        "./trading_bot/ensemble_models/eq_3301_ep_78.pt",
        "./trading_bot/ensemble_models/eq_3296_ep_60.pt", 
      ],
     "architecture": Model_3},
]



MODELS = []

def load_model():
    for index, model_info in enumerate(MODELS_INFOS):
        print(f"Loading model {index+1} ...")

        model_paths = model_info["paths"]
        architecture = model_info["architecture"]

        for model_path in model_paths:
            # Initialize model
            model = architecture()

            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.to('cpu')

            # Add model to the list
            MODELS.append(model)


def get_action(opens, highs, lows, closes, volumes, train_scalers):
    combined_actions = []

    # Build features
    features, num_features, train_scalers = build_features(opens, highs, lows, closes, volumes, train_scalers)

    x = np.expand_dims(features, axis=0)
    x = torch.from_numpy(x.astype(np.float32))

    with torch.no_grad():
        for model in MODELS:
            model.train(False)

            y_pred = model(x).detach().cpu().numpy()

            action = y_pred[0, 0] > 0 # 0: Sell, 1: Buy
            combined_actions.append(action)

    final_action = 1 if combined_actions.count(1) > combined_actions.count(0) else 0
    return final_action




