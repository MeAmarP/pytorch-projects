import torch
import torch.nn as nn
from torchvision.models import alexnet
from libcnn import AlexNet

def test_alexnet_layer_shapes():
    batch_size = 32
    num_classes = 10
    input_shape = (batch_size, 3, 224, 224)
    libcnn_model = AlexNet(num_classes)
    torchvision_model = alexnet(num_classes)
    
    x = torch.randn(input_shape)
    
    # Check convolutional layer shapes
    libcnn_conv_shapes = []
    for layer in libcnn_model.children():
        if isinstance(layer, nn.Sequential):
            for conv_layer in layer.children():
                if isinstance(conv_layer, nn.Conv2d):
                    print("libcnn: ", conv_layer.weight.shape)
                    libcnn_conv_shapes.append(tuple(conv_layer.weight.shape))
    torchvision_conv_shapes = []
    for layer in torchvision_model.features.children():
        if isinstance(layer, nn.Conv2d):
            print("torchvision: ",layer.weight.shape)
            torchvision_conv_shapes.append(tuple(layer.weight.shape))
    assert libcnn_conv_shapes == torchvision_conv_shapes, "Convolutional layer shapes do not match"
    
    # Check fully connected layer shapes
    libcnn_fc_shapes = []
    for layer in libcnn_model.children():
        if isinstance(layer, nn.Sequential):
            for fc_layer in layer.children():
                if isinstance(fc_layer, nn.Linear):
                    libcnn_fc_shapes.append(tuple(fc_layer.weight.shape))
    torchvision_fc_shapes = []
    for layer in torchvision_model.classifier.children():
        if isinstance(layer, nn.Linear):
            torchvision_fc_shapes.append(tuple(layer.weight.shape))
    assert libcnn_fc_shapes == torchvision_fc_shapes, "Fully connected layer shapes do not match"
    
    print("All layer shapes match!")
    
if __name__ == '__main__':
    try:
        test_alexnet_layer_shapes()
    except AssertionError as e:
        print(e)
        raise SystemExit(1)
