import torch
import torch.nn as nn

# TODO
# * Rewrite a standard CNN model layer-by-layer using PyTorch.
# * Compare model architecture with models in torchvision.
# * Use 'assert' method to confirm the output value/layer numbers with sample data input


# * LeNet5
# NOTE: This is modified LeNet5 Model which uses ReLU activation function instead of Tanh.
class LeNet5(nn.Module):
    """
    LeNet5 is a convolutional neural network architecture that is used for image classification tasks.
    This implementation has two convolutional layers followed by three fully connected layers.

    Args:
        num_classes (int): Number of classes in the classification task.

    Returns:
        torch.Tensor: The output tensor of the network.
    """

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.fc(x)
        return x


# Define a class of AlexNet CNN model
class AlexNet(nn.Module):
    """
    Implementation of AlexNet architecture.

    Args:
    - num_classes (int): number of classes in the classification task.

    Returns:
    - output tensor (torch.Tensor): tensor of shape (batch_size, num_classes) containing the predicted scores for each class.
    """

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.convlayer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        """
        Forward pass of the AlexNet model.

        Args:
        - x (torch.Tensor): input tensor of shape (batch_size, 3, height, width).

        Returns:
        - output tensor (torch.Tensor): tensor of shape (batch_size, num_classes) containing the predicted scores for each class.
        """
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.convlayer5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Define a class for VGG16 Model
class VGG16(nn.Module):
    """
    Implementation of VGG16 architecture.

    Args:
    - num_classes (int): number of classes in the classification task.

    Returns:
    - output tensor (torch.Tensor): tensor of shape (batch_size, num_classes) containing the predicted scores for each class.
    """

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.convlayer1 = self.vgg_block(3, 64, 2)
        self.convlayer2 = self.vgg_block(64, 128, 2)
        self.convlayer3 = self.vgg_block(128, 256, 3)
        self.convlayer4 = self.vgg_block(256, 512, 3)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            # nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def vgg_block(self, in_channels, out_channels, num_conv_layers):
        """
        Defines a VGG block consisting of multiple convolutional layers followed by a max pooling layer.

        Args:
        - in_channels (int): number of input channels.
        - out_channels (int): number of output channels.
        - num_conv_layers (int): number of convolutional layers in the block.

        Returns:
        - block (nn.Sequential): sequential container of the VGG block layers.
        """
        layers = []
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        block = nn.Sequential(*layers)
        return block
