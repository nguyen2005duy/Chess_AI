import torch
import torch.nn as nn
import torch.nn.functional as F
import chess


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow in deep networks"""

    def __init__(self, hidden_size, dropout_rate=0.3880649861019966):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass with residual connection"""
        identity = x

        # First linear layer
        out = self.linear1(x)
        if out.dim() > 1 and out.size(0) > 1:
            out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Second linear layer
        out = self.linear2(out)
        if out.dim() > 1 and out.size(0) > 1:
            out = self.bn2(out)

        # Add residual connection
        out += identity
        out = F.gelu(out)

        return out


class NNUE(nn.Module):
    def __init__(self, input_size=768, hidden_size=384, dropout_rate=0.3880649861019966):
        super(NNUE, self).__init__()

        # Feature transformer layers with improved structure
        self.ft_white = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),  # GELU activation often performs better than ReLU for chess eval
            nn.Dropout(dropout_rate)
        )

        self.ft_black = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Improved network backbone with residual connections
        self.hidden1 = nn.Linear(2 * hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Add a residual block for better gradient flow
        self.res_block = ResidualBlock(hidden_size, dropout_rate)

        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        # Output layer with proper scaling for evaluation
        self.output = nn.Linear(hidden_size // 2, 1)

        # Initialize weights with carefully chosen values
        self._init_weights()

    def _init_weights(self):
        """Smart weight initialization for chess evaluation networks"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Careful initialization improves convergence
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    # Small positive bias helps with piece values
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, white_features, black_features, stm=None):
        # Transform features from both perspectives
        white_transformed = self.ft_white(white_features)
        black_transformed = self.ft_black(black_features)

        # Concatenate with side-to-move awareness
        if stm is not None:
            # Convert float tensor to boolean tensor if needed
            if stm.dtype != torch.bool:
                stm_bool = stm > 0.5
            else:
                stm_bool = stm

            # If side to move is provided, use it to order the features
            # This helps the network learn move-specific patterns
            x = torch.cat([
                torch.where(stm_bool.unsqueeze(1), white_transformed, black_transformed),
                torch.where(stm_bool.unsqueeze(1), black_transformed, white_transformed)
            ], dim=1)
        else:
            # Standard concatenation
            x = torch.cat([white_transformed, black_transformed], dim=1)

        # Enhanced forward pass with residual connections
        x = self.hidden1(x)
        if x.dim() > 1 and x.size(0) > 1:
            x = self.bn1(x)
        x = F.gelu(x)

        # Apply residual block
        x = self.res_block(x)

        # Second hidden layer
        x = self.hidden2(x)
        if x.dim() > 1 and x.size(0) > 1:
            x = self.bn2(x)
        x = F.gelu(x)

        # Output scaled to match the input scale (divided by 10 in converter)
        # Change this line in your NNUE forward method:
        return torch.tanh(self.output(x)) * 1000  # Reduced from 1000

    def save(self, filepath):
        """Save the NNUE model with metadata"""
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.ft_white[0].in_features,
            'hidden_size': self.ft_white[0].out_features,
            'dropout_rate': self.ft_white[2].p,
            'version': '2.0'  # Track versions for compatibility
        }, filepath)

    def load(self, filepath):
        """Load the NNUE model with error handling (instance method)"""
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_state_dict(checkpoint)
            self.eval()
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    @classmethod
    def load(cls, filepath):
        """Class method for loading an NNUE model from a file"""
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Extract model parameters if they exist in the checkpoint
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Get parameters from saved metadata if available
                input_size = checkpoint.get('input_size', 768)
                hidden_size = checkpoint.get('hidden_size', 384)
                dropout_rate = checkpoint.get('dropout_rate', 0.3880649861019966)

                # Create a new model with the extracted parameters
                model = cls(input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # If no metadata, use default parameters
                model = cls()
                model.load_state_dict(checkpoint)

            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None