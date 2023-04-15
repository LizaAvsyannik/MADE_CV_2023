from torch import nn
import torchvision.models as _M


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


class SwinSmallPatched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.swin_s(_M.Swin_S_Weights, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.head = nn.Sequential(
            nn.Linear(768, self._n_classes),
            nn.LogSoftmax(),
        )
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return list(self._model.features.modules())[-6:] + [
            self._model.norm,
        ]
    
    def _get_transfer_trainable_layers(self):
        return [self._model.head]

    @staticmethod
    def get_input_transform():
        return _M.Swin_S_Weights.DEFAULT.transforms()

    def forward(self, x):
        return self._model(x)


class ResNet101Patched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.resnet101(_M.ResNet101_Weights, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.fc = nn.Sequential(
            nn.Linear(2048, self._n_classes),
            nn.LogSoftmax(),
        )
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return list(self._model.layer4.children())[-1:]
    
    def _get_transfer_trainable_layers(self):
        return [self._model.fc]

    @staticmethod
    def get_input_transform():
        return _M.ResNet101_Weights.DEFAULT.transforms()

    def forward(self, x):
        return self._model(x)


class SwinV2SmallPatched(nn.Module, TransferLearningMixin):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self._model = _M.swin_s(_M.Swin_V2_S_Weights, progress=True)
        self._model.requires_grad_(False)
        self._model.num_classes = self._n_classes
        self._model.head = nn.Sequential(
            nn.Linear(768, self._n_classes),
            nn.LogSoftmax(),
        )
        self._defrost_parameters()

    def _get_backbone_trainable_layers(self):
        return list(self._model.features.modules())[-6:] + [
            self._model.norm,
        ]
    
    def _get_transfer_trainable_layers(self):
        return [self._model.head]

    @staticmethod
    def get_input_transform():
        return _M.Swin_V2_S_Weights.DEFAULT.transforms()

    def forward(self, x):
        return self._model(x)


MODEL_NAME_TO_CLASS = {
    'swin_s': SwinSmallPatched,
    'swin_v2_s': SwinV2SmallPatched,
    'resnet101': ResNet101Patched,
}


if __name__ == '__main__':
    model = SwinV2SmallPatched(30)
    from torchsummary import summary

    summary(model, (3, 300, 300), batch_size=256, device='cpu')
    print(model._get_backbone_trainable_layers())
    print(model._get_transfer_trainable_layers())
