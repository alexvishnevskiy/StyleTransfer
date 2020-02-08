from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from pathlib import Path
import copy


def Gramm_matrix(matrix):
  a, b, c, d = matrix.size() #b-количество фичей; с, d-размер  
  matrix = matrix.view(a*b, c*d)
  matrix = torch.mm(matrix, matrix.t())
  return matrix.div(a*b*c*d)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    
    
class Content_Loss(nn.Module):

  def __init__(self, target):
    super(Content_Loss, self).__init__()
    self.target = target.detach()
  
  def forward(self, input):
    self.loss = F.mse_loss(self.target, input)
    return input


class Style_Loss(nn.Module):

  def __init__(self, style):
    super(Style_Loss, self).__init__()
    self.style = Gramm_matrix(style).detach()

  def forward(self, input):
    self.loss = F.mse_loss(self.style, Gramm_matrix(input))
    return input


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


class StyleTransferModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v2(pretrained=True).features.to(self.device).eval()
        self.imsize = 512
        self.loader = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),  # нормируем размер изображения
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.normalization = Normalization(self.mean, self.std)
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        
    def transfer_style(self, content_img_stream, style_img_stream, num_steps=100, style_weight=1000000, content_weight=1):
        target_img = self.image_loader(content_img_stream)
        style_img = self.image_loader(style_img_stream)
        input_img = target_img.clone()
        optimizer = get_input_optimizer(input_img)
        model, style_losses, content_losses = self.get_style_and_loss(style_img, target_img, self.content_layers_default, self.style_layers_default)
        run = [0]
        while num_steps> run[0]:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()  
                run[0] += 1
                return style_score + content_score
            optimizer.step(closure)    
            input_img.data.clamp_(0, 1)
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # В первой итерации, когда вы переносите уже готовую модель из тетрадки с занятия сюда нужно просто
        # перенести функцию run_style_transfer (не забудьте вынести инициализацию, которая
        # проводится один раз в конструктор.

        # Сейчас этот метод просто возвращает не измененную content картинку
        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        return self._unloader(input_img)

    # В run_style_transfer используется много внешних функций, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # Эта функция тоже не является
    def get_style_and_loss(self, style_img, target_img, content_layers, style_layers):
        cnn = copy.deepcopy(self.model)
        content_losses = []
        style_losses = []
        model = nn.Sequential(self.normalization)
        i = 0
        for layer in cnn.children():
            i += 1
            name = 'conv_{}'.format(i)
            model.add_module(name, layer)
            
            if name in content_layers:
                target = model(target_img).detach()
                content_loss = Content_Loss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = Style_Loss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
  
        for j in range(len(model) - 1, -1, -1):
            if isinstance(model[j], Content_Loss) or isinstance(model[j], Style_Loss):
                break

        model = model[:(j + 1)]

        return model, style_losses, content_losses
    
    def _unloader(self, tensor):
        transform = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)      # remove the fake batch dimension
        image = transform(image)
        return image
        
    def image_loader(self, img_stream):
        image = Image.open(img_stream)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
