# Importing useful libraries
import torch
import numpy as np
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models


'''
In PyTorch VGG is split into two parts:

vgg19.features, which contains all the convolutional and pooling layers
vgg19.classifier, which contains all the three linear classifier layers
'''

vgg = models.vgg19(pretrained=True).features

# freeze all the VGG parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Moving the module to CPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)


def load_image(img_path, max_size=400, shape=None):
    """
    This method loads in and transforms an image, making sure the image is
    smaller than or equal to the max_size
    """
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


content = load_image('/Users/siddhantbansal/Desktop/tqdm.jpeg').to(device)
style = load_image('/Users/siddhantbansal/Desktop/Tsunami_by_hokusai_19th_century.jpg').to(device)


def im_convert(tensor):
    """
    Helper function for un-normalizing an image
    This function converts the tensor to Numpy Image for display
    """
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

print(vgg)


def get_features(image, model, layers=None):
    """
    This method is used to get the features of an input image through VGG19
    """
    if layers is None:
        layers = {'0': 'conv1_1',
                  '2': 'conv1_2',
                  '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '12': 'conv3_2',
                  '14': 'conv3_3',
                  '16': 'conv3_4',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '23': 'conv4_3',
                  '25': 'conv4_4',
                  '28': 'conv5_1',
                  '30': 'conv5_2',
                  '32': 'conv5_3',
                  '34': 'conv5_4'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """
    This method calculates the Gram matrix of a given tensor
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram


# get content and style features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrix for each layer of the style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third 'target' image and prep if for a change
# It's a good idea to start with the target as a copy of the content image
# then iteratively changing it's style
target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

content_weight = 1  # alpha
style_weight = 1e6  # beta

show_every = 1

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 500
print('Starting the main loop')
for ii in range(1, steps+1):
    print('Loop Number ', ii)
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_weights:
        # get the target style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        target_gram = gram_matrix(target_feature)

        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()
