import torch
import torch.nn as nn


class WordWindowClassifier(torch.nn.Module):
    def __init__(
            self,
            window_size: int,
            embed_dim: int,
            hidden_dim: int,
            freeze_embeddings: bool,
            vocab_size: int,
            pad_idx: int,
    ):
        super().__init__()

        self.window_size = window_size

        # Embedding layer - maps indices to embeddings
        self.embeds = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Freeze weights if freeze_embeddings = True
        if freeze_embeddings:
            self.embeds.weight.requires_grad = False

        full_window_size = 2 * self.window_size + 1
        # Linear(full_window_size * embed_dim, hidden_dim) -> Tanh
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * embed_dim, hidden_dim),
            nn.Tanh()
        )

        # Linear -> Sigmoid
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B x L], B - batch_size, L - sequence length
        """

        # Unfold inputs into sequence of windows of length `2 * self.window_size + 1`
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)

        """
        token_windows: [B x L~ x full_window_size]
        full_window_size = 2 * self.window_size + 1
        L~ = L - (2 * self.window_size + 1) + 1
        """

        # Embed token_windows into embeddings
        embedded_windows = self.embeds(token_windows)
        embedded_windows = embedded_windows.view(embedded_windows.size(0), embedded_windows.size(1), -1)

        """
        embedded_windows: [B x L~ x (full_window_size * EMBED_DIM)]
        """

        # Pass embedded_windows into hidden_layer
        layer_1 = self.hidden_layer(embedded_windows)

        """
        layer_1: [B x L~ x HIDDEN_DIM]
        """

        # Pass layer_1 into output_layer
        output = self.output_layer(layer_1)
        output = output.view(output.size(0), output.size(1))

        """
        output: [B x L~]
        """

        return output
