import torch
import torch.nn.functional as F
import processing


INPUT_SCALED_INT = 1
INPUT_EQUALIZED_INT = 2
INPUT_NORMALIZED_INT = 4
INPUT_NORMALIZED_GRADIENTS_OLD = 8
INPUT_NORMALIZED_GRADIENTS = 16


_INPUT_CHANNELS = {
    INPUT_SCALED_INT: 1,
    INPUT_EQUALIZED_INT: 1,
    INPUT_NORMALIZED_INT: 1,
    INPUT_NORMALIZED_GRADIENTS_OLD: 6,
    INPUT_NORMALIZED_GRADIENTS: 6
}


_INPUT_CODES = {
    "s": INPUT_SCALED_INT,
    "e": INPUT_EQUALIZED_INT,
    "n": INPUT_NORMALIZED_INT,
    "d": INPUT_NORMALIZED_GRADIENTS_OLD,
    "g": INPUT_NORMALIZED_GRADIENTS
}


OUTPUT_RGB = 1
OUTPUT_GRADIENT_MAGNITUDES = 2
OUTPUT_GRADIENT_MAGNITUDES_D4 = 4


_OUTPUT_CODES = {
    "c": OUTPUT_RGB,
    "m": OUTPUT_GRADIENT_MAGNITUDES,
    "M": OUTPUT_GRADIENT_MAGNITUDES_D4,
}


def _input_from_code(code):
    return sum(_INPUT_CODES[c] for c in code)


def _output_from_code(code):
    return sum(_OUTPUT_CODES[c] for c in code)


class BaseCCNet(torch.nn.Module):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, noise=100.0, input_code="sd", output_code="c", mask_clipped=False):
        super().__init__()
        self.noise = noise
        self.mask_clipped = mask_clipped
        self.input_channels = 0
        self.output_channels = len(output_code)
        self.input = _input_from_code(input_code)
        self.output = _output_from_code(output_code)
        for k, v in _INPUT_CHANNELS.items():
            if k & self.input:
                self.input_channels += v

    def preprocessing(self, rgb):
        """Given the image, compute the actual imput to the CNN."""
        data = []
        if self.input & INPUT_SCALED_INT:
            gray = processing.scaled_intensity(rgb)
            data.append(2 * gray - 1)
        if self.input & INPUT_EQUALIZED_INT:
            gray = torch.mean(rgb, 1, keepdim=True)
            gray = processing.histogram_equalization(gray, bins=64)
            data.append(2 * gray - 1)
        if self.input & INPUT_NORMALIZED_INT:
            gray = torch.mean(rgb, 1, keepdim=True)
            gray = F.instance_norm(gray)
            data.append(gray)
        if self.input & INPUT_NORMALIZED_GRADIENTS_OLD:
            gradients = processing.spatial_gradients(rgb)
            directions = processing.normalize_gradients(gradients)
            data.append(2 * directions - 1)
        if self.input & INPUT_NORMALIZED_GRADIENTS:
            gradients = processing.spatial_gradients(rgb)
            directions = processing.normalize_gradients(gradients)
            data.append(directions)
        if len(data) == 1:
            return data[0]
        else:
            return torch.cat(data, 1)

    def compute_rgb_data(self, rgb):
        """Given the image, compute the color data to combine to get the estimate."""
        data = []
        if self.output & OUTPUT_RGB:
            data.append(rgb)
        if self.output & OUTPUT_GRADIENT_MAGNITUDES:
            grad = processing.spatial_gradients(rgb)
            data.append(processing.gradient_magnitude(grad))
        if self.output & OUTPUT_GRADIENT_MAGNITUDES_D4:
            grad = processing.spatial_gradients(rgb)
            data.append(processing.gradient_magnitude(grad) / 4.0)
        if len(data) == 1:
            return data[0].unsqueeze(2)
        else:
            return torch.stack(data, 2)

    def make_estimate(self, rgb, weights, noise=None, mask_clipped=None):
        """Compute the estimate given rgb and weights.

        rgb: Bx3xHxW
        weights: BxCxHxW

        result: Bx3
        """
        if mask_clipped is None:
            mask_clipped = self.mask_clipped
        if mask_clipped:
            ma = rgb.max(1, keepdim=True)[0]
            mm = ma.view(ma.size(0), 1, -1).max(2)[0].unsqueeze(-1).unsqueeze(-1)
            weights = weights * (ma < mm).float()
        data = self.compute_rgb_data(rgb)
        data = data.view(data.size(0), 3, -1)
        weights = weights.view(weights.size(0), 1, -1)
        estimate = torch.sum(data * weights, 2)
        if self.training:
            if noise is None:
                noise = self.noise
            estimate = estimate + noise * torch.randn_like(estimate)
        return torch.nn.functional.normalize(estimate)

    def last_parameters(self):
        """PArameters to be used for transfer learning."""
        return []



def conv_module(in_channels, out_channels, batch_normalization=True, relu=True):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if batch_normalization:
        layers.append(torch.nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(torch.nn.LeakyReLU(0.2))
    return torch.nn.Sequential(*layers)


def deconv_module(in_channels, out_channels, dropout=False, batch_normalization=True, relu=True):
    layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if batch_normalization:
        layers.append(torch.nn.BatchNorm2d(out_channels))
    if dropout:
        layers.append(torch.nn.Dropout2d())
    if relu:
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def id_layer(x):
    return x


class UNetModule(torch.nn.Module):
    """Compute a BxIxHxW -> BxOxHxW transform.

    The module performs a convolutional block, a deconvolutional block
    and optionally another module in the middle.

    When skip is False the number of output channels O is the same of
    the number of input channels I.  When skip is True O is double of
    I.

    When not None the inner module is expected to keep the image
    dimensions, keeping also constant the number of channels when skip
    is False, or doubling it when skip is True.

    """
    def __init__(self, channels, mid_channels, inner_module=None, skip=True, dropout=False):
        super().__init__()
        self.down = conv_module(channels, mid_channels)
        self.inner = (inner_module if inner_module is not None else id_layer)
        inner_channels = 2 * mid_channels if inner_module is not None and skip else mid_channels
        self.up = deconv_module(inner_channels, channels, dropout=dropout)
        self.skip = skip
        
    def forward(self, x):
        y = self.up(self.inner(self.down(x)))
        if self.skip:
            y = torch.cat([x, y], 1)
        return y

    def decoder_parameters(self):
        p = list(self.up.parameters())
        if self.inner is not id_layer:
            p.extend(self.inner.decoder_parameters())
        return p

    
class CCNet(BaseCCNet):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, skip=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv_module(self.input_channels, 64,
                                batch_normalization=False)
        m = UNetModule(512, 512, None, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, skip=skip)
        m = UNetModule(256, 512, m, skip=skip)
        m = UNetModule(128, 256, m, skip=skip)
        m = UNetModule(64, 128, m, skip=skip)
        self.inner = m
        channels = (128 if skip else 64)
        self.deconv = deconv_module(channels, self.output_channels,
                                    batch_normalization=False,
                                    relu=False)

    def forward(self, rgb):
        x = self.preprocessing(rgb)
        logits = self.deconv(self.inner(self.conv(x)))
        weights = torch.sigmoid(logits)
        estimate = self.make_estimate(rgb, weights)
        return estimate, x, weights

    def last_parameters(self):
        # p = self.inner.decoder_parameters()
        # p.extend(self.deconv.parameters())
        # return p
        return self.deconv.parameters()


class SmallCCNet(BaseCCNet):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, skip=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv_module(self.input_channels, 32,
                                batch_normalization=False)
        m = UNetModule(256, 256, None, dropout=True, skip=skip)
        m = UNetModule(256, 256, m, skip=skip)
        m = UNetModule(128, 256, m, skip=skip)
        m = UNetModule(64, 128, m, skip=skip)
        m = UNetModule(32, 64, m, skip=skip)
        self.inner = m
        channels = (64 if skip else 32)
        self.deconv = deconv_module(channels, self.output_channels,
                                    batch_normalization=False,
                                    relu=False)

    def forward(self, rgb):
        x = self.preprocessing(rgb)
        logits = self.deconv(self.inner(self.conv(2 * x - 1)))
        weights = torch.sigmoid(logits)
        estimate = self.make_estimate(rgb, weights)
        return estimate, x, weights

    def last_parameters(self):
        return self.deconv.parameters()


def _test():
    net = CCNet(input_code="g", output_code="c", mask_clipped=True)
    print(net)
    x = torch.rand(2, 3, 256, 256)
    y = net(x)[0]
    print(x.size(), "->", y.size())


if __name__ == "__main__":
    _test()
