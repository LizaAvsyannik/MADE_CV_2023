import typing as tp

from math import log2, ceil

import torch
from torch import nn, optim
import torchvision.models as _M

from torchvision import transforms as T

from torchinfo import summary

from utils import MAX_OCR_LEN, TrainConfig


MAX_PREDICTED_TOKENS = 2 * MAX_OCR_LEN


class InputTransform:
    def __init__(self, resize_size):
        mean: tp.Tuple[float, ...] = (0.485, 0.456, 0.406)
        std: tp.Tuple[float, ...] = (0.229, 0.224, 0.225)
        self.resize_size = resize_size
        self._transform = T.Compose([
            T.Resize([resize_size, resize_size], interpolation=T.InterpolationMode.BICUBIC),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])

    @property
    def transform(self):
        return self._transform


class TransferLearningMixin:
    def _get_backbone_trainable_layers(self):
        raise NotImplementedError()

    def _get_transfer_trainable_layers(self):
        raise NotImplementedError()
    
    def _defrost_parameters(self):
        for module in self._get_backbone_trainable_layers():
            module.requires_grad_(True)
        for module in self._get_transfer_trainable_layers():
            module.requires_grad_(True)

    def get_backbone_trainable_parameters(self):
        for module in self._get_backbone_trainable_layers():
            yield from module.parameters()

    def get_transfer_trainable_parameters(self):
        for module in self._get_transfer_trainable_layers():
            yield from module.parameters()


class ConvBlock(nn.Module):
    def __init__(
        self, n_layers, input_ch, output_ch, kernel_size,
        conv_layer=nn.Conv2d, activation=nn.ReLU, **kwargs
    ):
        super().__init__()
        kwargs.setdefault('padding', 'same')

        conv_activ = lambda **conv_kwargs: nn.Sequential(conv_layer(**conv_kwargs), activation())

        layers = [conv_activ(in_channels=input_ch, out_channels=output_ch, kernel_size=kernel_size, **kwargs)]
        repeated_convs_kwargs = {'in_channels': output_ch, 'out_channels': output_ch, 'kernel_size': kernel_size}
        repeated_convs_kwargs.update(kwargs)
        for i in range(n_layers):
            layers.append(conv_activ(**repeated_convs_kwargs))
        self._block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._block(x)


class RNNOCRHead(nn.Module):
    def __init__(
        self, embedding_size: int, max_ocr_len: int, vocab_size: int,
    ):
        super().__init__()
        self._n_patches = 2 ** ceil(log2(max_ocr_len))
        assert embedding_size % self._n_patches == 0
        self._patch_size = embedding_size // self._n_patches

        self._cnn_expand = nn.Sequential(
            ConvBlock(3, 1, 8, 7, conv_layer=nn.Conv1d),
            ConvBlock(3, 8, 32, 5, conv_layer=nn.Conv1d),
            ConvBlock(3, 32, 64, 5, conv_layer=nn.Conv1d),
            ConvBlock(3, 64, self._n_patches, 3, conv_layer=nn.Conv1d),
        )
        self._lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=384, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )
        self._fc = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, vocab_size),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self._cnn_expand(x)
        x = x.view(x.shape[0], self._n_patches, self._n_patches, self._patch_size)
        x = x.permute(0, 2, 1, 3)
        x = nn.Flatten(2)(x)
        encoded, (hn, cn) = self._lstm(x)
        return self._fc(encoded)


class SimpleRNNOCRHead(nn.Module):
    def __init__(
        self, embedding_size: int, vocab_size: int,
    ):
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=384, num_layers=2,
            bidirectional=True,
        )
        self._fc = nn.Sequential(
            nn.Linear(768, vocab_size),
        )

    def forward(self, x: torch.Tensor):
        encoded, _ = self._lstm(x)
        return self._fc(encoded).log_softmax(-1)


class SwinSmallPatched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.swin_s(weights=_M.Swin_S_Weights.DEFAULT, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.head = RNNOCRHead(768, MAX_PREDICTED_TOKENS, n_classes)
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return list(self._model.features.modules())[-6:] + [
            self._model.norm,
        ]
    
    def _get_transfer_trainable_layers(self):
        return [self._model.head]

    @staticmethod
    def get_input_transform():
        # return _M.Swin_S_Weights.DEFAULT.transforms()
        return InputTransform(256).transform

    def forward(self, x):
        return self._model(x)


class ResNet18Patched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._backbone = _M.resnet18(weights=_M.ResNet18_Weights.DEFAULT, progress=True)
        self._backbone = nn.Sequential(*list(self._backbone.children())[:-2])
        self._backbone.requires_grad_(True)
        self._num_output_features = self._backbone[-1][-1].bn2.num_features
        with torch.no_grad():
            size = 224
            latent_space_width = self._backbone(torch.randn(size=(10, 3, size, size))).shape[3]
        
        self._pool = nn.AdaptiveAvgPool2d((1, latent_space_width))
        self._proj = ConvBlock(3, latent_space_width, 2 * MAX_OCR_LEN, 3)
        self._decoder = SimpleRNNOCRHead(self._num_output_features, n_classes)
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return [self._backbone]
    
    def _get_transfer_trainable_layers(self):
        return [self._proj, self._decoder]

    @staticmethod
    def get_input_transform():
        # return _M.ResNet18_Weights.DEFAULT.transforms()
        return InputTransform(224).transform
    
    def _apply_projection(self, x):
        """Use convolution to increase width of a features.

        Args:
            - x: Tensor of features (shaped B x C x H x W).

        Returns:
            New tensor of features (shaped W' x B x C).
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self._proj(x)
        x = x.squeeze(2).permute(1, 0, 2)
        return x

    def forward(self, x):
        x = self._backbone(x)
        x = self._pool(x)
        x = self._apply_projection(x)
        return self._decoder(x)


class ResNet101Patched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.resnet101(weights=_M.ResNet101_Weights.DEFAULT, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.fc = RNNOCRHead(2048, MAX_PREDICTED_TOKENS, n_classes)
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return [self._model.layer4]
    
    def _get_transfer_trainable_layers(self):
        return [self._model.fc]

    @staticmethod
    def get_input_transform():
        # return _M.ResNet101_Weights.DEFAULT.transforms()
        return InputTransform(224).transform

    def forward(self, x):
        return self._model(x)


class SwinV2SmallPatched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.swin_v2_s(weights=_M.Swin_V2_S_Weights.DEFAULT, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.head = RNNOCRHead(768, MAX_PREDICTED_TOKENS, n_classes)
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return list(self._model.features.modules())[-6:] + [
            self._model.norm,
        ]

    def _get_transfer_trainable_layers(self):
        return [self._model.head]

    @staticmethod
    def get_input_transform():
        # return _M.Swin_V2_S_Weights.DEFAULT.transforms()
        return InputTransform(256).transform

    def forward(self, x):
        return self._model(x)


MODEL_NAME_TO_CLASS = {
    'swin_s': SwinSmallPatched,
    'swin_v2_s': SwinV2SmallPatched,
    'resnet101': ResNet101Patched,
    'resnet18': ResNet18Patched,
}


def load_model(
    checkpoint_path: str, device: torch.device,
    model: nn.Module=None, optimizer: optim.Optimizer=None, scheduler: optim.lr_scheduler._LRScheduler=None,
):
    state_dict = torch.load(checkpoint_path, map_location=device)
    train_config = TrainConfig(**state_dict['config'])
    train_config.device = device
    if model is None:
        model = MODEL_NAME_TO_CLASS[train_config.backbone](state_dict['n_classes'])
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])

    return model.to(device), train_config


if __name__ == '__main__':
    model = MODEL_NAME_TO_CLASS['resnet18'](600)

    summary(model, input_size=(64, 3, 300, 300), device='cpu', depth=5)
    print(model._get_backbone_trainable_layers())
    print(model._get_transfer_trainable_layers())
