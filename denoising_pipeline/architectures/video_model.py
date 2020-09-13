import torch
import torch.nn as nn
from denoising_pipeline.architectures.resnet_block import BasicBlock


class OneImageNetModel(nn.Module):
    def __init__(self, output_filters=3,
                 residuals=True, n=1, activation=nn.ReLU()):
        super(OneImageNetModel, self).__init__()

        self.conv1 = nn.Conv2d(
            3, output_filters, kernel_size=5, bias=False, stride=1
        )

        self.conv_list = []
        for i in range(n):
            if residuals:
                self.conv_list.append(
                    BasicBlock(output_filters, output_filters)
                )
            else:
                self.conv_list.append(
                    nn.Conv2d(
                        output_filters, output_filters,
                        kernel_size=5, bias=False
                    )
                )
                self.conv_list.append(activation)
        self.conv_list = nn.Sequential(*tuple(self.conv_list))

        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv_list(x)
        return x


class VideoImprovingNet(nn.Module):
    def __init__(self, n=1,
                 residuals=True, series_size=6, filters_per_image=32):
        super(VideoImprovingNet, self).__init__()

        self.input_images_count = series_size
        self.filters_per_model = filters_per_image

        self.per_image_models = nn.ModuleList([
            OneImageNetModel(
                output_filters=filters_per_image,
                residuals=residuals,
                n=n
            )
            for i in range(series_size)
        ])

        self.conv1 = nn.Conv2d(
            filters_per_image * series_size,
            32,
            kernel_size=5,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            32,
            16,
            kernel_size=5,
            bias=False
        )

        self.conv_out = nn.Conv2d(
            16,
            3,
            kernel_size=1,
            bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, *input):
        assert len(input) == len(self.per_image_models)
        base_models_outputs = []
        for i, input_image in enumerate(input):
            base_models_outputs.append(
                self.per_image_models[i](input_image)
            )

        x = torch.cat(tuple(base_models_outputs), dim=1)

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv_out(x)

        return x

    def _inference(self, x):
        x = torch.cat(
            (self.per_image_models[0](x),) * self.input_images_count,
            dim=1
        )

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv_out(x)

        return torch.sigmoid(x)

    def inference(self, x):
        return self.forward(*tuple([
                               x
                           ] * self.input_images_count))
