from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import max_pool2d, relu
from torch import Tensor

class SimpleCNN(Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc_out = Linear(in_features=128, out_features=10)


    
    def forward(self, x: Tensor, verbose=False) -> Tensor:

        if verbose: print(f"x.shape input: {x.shape}")
        
        x = self.conv1(x)
        if verbose: print(f"x.shape after conv1: {x.shape}")
        x = relu(x)
        if verbose: print(f"x.shape after relu1: {x.shape}")
        x = max_pool2d(x, 2, 2)
        if verbose: print(f"x.shape after maxpool1: {x.shape}")

        x = self.conv2(x)
        if verbose: print(f"x.shape after conv2: {x.shape}")
        x = relu(x)
        if verbose: print(f"x.shape after relu2: {x.shape}")
        x = max_pool2d(x, 2, 2)
        if verbose: print(f"x.shape after maxpool2: {x.shape}")

        x = self.conv3(x)
        if verbose: print(f"x.shape after conv3: {x.shape}")
        x = relu(x)
        if verbose: print(f"x.shape after relu3: {x.shape}")
        x = max_pool2d(x, 2, 2)
        if verbose: print(f"x.shape after maxpool3: {x.shape}")

        x = x.reshape(-1, 128 * 1 * 1)
        if verbose: print(f"x.shape after reshape: {x.shape}")

        x = self.fc_out(x)
        if verbose: print(f"x.shape after fc_out: {x.shape}")

        return x



def main():

    import torch

    x = torch.rand((128, 1, 28, 28))

    model = SimpleCNN()

    print(model)

    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    y_logits = model.forward(x, True)

    print(f"y_logits.shape: {y_logits.shape}")


if __name__ == "__main__":
    main()