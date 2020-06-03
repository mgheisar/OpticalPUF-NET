import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class modelTriplet(nn.Module):
    """Constructs a model using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, model_architecture="resnet50", pretrained=False):
        super(modelTriplet, self).__init__()

        if model_architecture == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            # self.model.avgpool = nn.AdaptiveAvgPool2d(1)  # when input size is not 224*224
            input_features_fc_layer = self.model.fc.in_features
            # Output embedding
            self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

        elif model_architecture == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
            input_features_fc_layer = self.model.classifier[-1].in_features
            mod = list(self.model.classifier.children())
            mod.pop()
            mod.append(nn.Linear(input_features_fc_layer, embedding_dimension))
            new_classifier = nn.Sequential(*mod)
            self.model.classifier = new_classifier

        elif model_architecture == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

        elif model_architecture == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

        elif model_architecture == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

        elif model_architecture == "inceptionresnetv2":
            self.model = models.inceptionresnetv2(pretrained=pretrained)
            self.model.last_linear = nn.Linear(1536, embedding_dimension)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        # print(next(self.model.parameters()).is_cuda)
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #   Equation 9: number of classes C, P=0.9
        #   lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding
