import torch
import torch.nn as nn


class GLU(nn.Module):
    """
    Gated Linear Unit.

    GLU divides the input features into two parts and applies an element-wise
    product between the first part and a sigmoid of the second part.

    Formula: GLU(x) = x_1 ⊗ σ(x_2)
    where x_1 and x_2 are split from x along the feature dimension.
    """

    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        dim: int = -1,
        nonlinearity: str = "sigmoid",
    ) -> None:
        """
        Initialize a GLU layer.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features. If None, output_dim = input_dim // 2
            bias: Whether to use bias in the linear layer
            dim: Dimension along which to split the input
            nonlinearity: Nonlinearity to use for the gate ('sigmoid' or 'tanh')
        """
        super(GLU, self).__init__()
        self.input_dim = input_dim

        self.projection = nn.Linear(input_dim, input_dim * 2, bias=bias)

        # Dimension to split on
        self.dim = dim

        # Gate nonlinearity
        if nonlinearity.lower() == "sigmoid":
            self.gate_nonlinearity = torch.sigmoid
        elif nonlinearity.lower() == "tanh":
            self.gate_nonlinearity = torch.tanh
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU to input tensor.

        Args:
            x: Input tensor of shape [..., input_dim, ...]

        Returns:
            Output tensor of shape [..., output_dim, ...]
        """
        # Project input if needed
        x = self.projection(x)

        # Split the tensor along specified dimension
        a, b = torch.split(x, self.input_dim, dim=self.dim)
        # Apply gate mechanism
        return a * self.gate_nonlinearity(b)


if __name__ == "__main__":
    batch_size = 16
    frame_size = 256
    input_shape = (batch_size, 1, frame_size)
    input = torch.randn(input_shape)
    print(f"input: {input}")
    glu = GLU(input_dim=frame_size)
    output = glu(input)
    print(f"output: {output}")
