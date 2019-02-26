import torch


def apply_map(data, vmin=None, vmax=None, dim=1, cmap="plasma"):
    """Make a color image from a tensor.

    - data: input data of size
    - vmin: smaller value to be represented
    - vmax: largest value to be represented
    - dim: index of the new color dimension
    - cmap: a colormap specified by name or by rgb data

    The size of the output will be the same of the input, but with an
    additional dimension of size 3.  If vmin and vmax are not
    specified the minimum and the maximum value in data are used.

    """
    data = data.float()
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    if isinstance(cmap, str):
        cmap = COLORMAPS[cmap].to(data.device)
    n = cmap.size(0)
    x = (n - 1) * (data.float() - vmin) / (vmax - vmin)
    indices = torch.clamp(x.long(), 0, n - 2)
    frac = torch.clamp(x - indices.float(), 0, 1).unsqueeze(-1)
    try:
        image0 = cmap[indices]
        image1 = cmap[indices + 1]
        image = image0 * (1 - frac) + image1 * frac
    except Exception as e:
        print(e)  # This should never happen, but sometimes... (maybe a bug?)
        image = torch.zeros_like(frac)
    # Rotate the last dimension until it reaches the parameter 'dim'
    for d in range(image.dim())[-1:dim:-1]:
        image = torch.transpose(image, d - 1, d)
    return image


# The next four are downsamples of the new matplotlib colormaps by
# Nathaniel J. Smith, Stefan van der Walt, and (in the case of
# viridis) Eric Firing.

_viridis = [
    [0.267004, 0.004874, 0.329415],
    [0.282656, 0.100196, 0.422160],
    [0.277134, 0.185228, 0.489898],
    [0.253935, 0.265254, 0.529983],
    [0.221989, 0.339161, 0.548752],
    [0.190631, 0.407061, 0.556089],
    [0.163625, 0.471133, 0.558148],
    [0.139147, 0.533812, 0.555298],
    [0.120565, 0.596422, 0.543611],
    [0.134692, 0.658636, 0.517649],
    [0.208030, 0.718701, 0.472873],
    [0.327796, 0.773980, 0.406640],
    [0.477504, 0.821444, 0.318195],
    [0.647257, 0.858400, 0.209861],
    [0.824940, 0.884720, 0.106217],
    [0.993248, 0.906157, 0.143936]
]


_magma = [
    [0.001462, 0.000466, 0.013866],
    [0.043830, 0.033830, 0.141886],
    [0.123833, 0.067295, 0.295879],
    [0.232077, 0.059889, 0.437695],
    [0.341482, 0.080564, 0.492631],
    [0.445163, 0.122724, 0.506901],
    [0.550287, 0.161158, 0.505719],
    [0.658483, 0.196027, 0.490253],
    [0.767398, 0.233705, 0.457755],
    [0.868793, 0.287728, 0.409303],
    [0.944006, 0.377643, 0.365136],
    [0.981000, 0.498428, 0.369734],
    [0.994738, 0.624350, 0.427397],
    [0.997228, 0.747981, 0.516859],
    [0.993170, 0.870024, 0.626189],
    [0.987053, 0.991438, 0.749504]
]


_inferno = [
    [0.001462, 0.000466, 0.013866],
    [0.046915, 0.030324, 0.150164],
    [0.142378, 0.046242, 0.308553],
    [0.258234, 0.038571, 0.406485],
    [0.366529, 0.071579, 0.431994],
    [0.472328, 0.110547, 0.428334],
    [0.578304, 0.148039, 0.404411],
    [0.682656, 0.189501, 0.360757],
    [0.780517, 0.243327, 0.299523],
    [0.865006, 0.316822, 0.226055],
    [0.929644, 0.411479, 0.145367],
    [0.970919, 0.522853, 0.058367],
    [0.987622, 0.645320, 0.039886],
    [0.978806, 0.774545, 0.176037],
    [0.950018, 0.903409, 0.380271],
    [0.988362, 0.998364, 0.644924]
]


_plasma = [
    [0.050383, 0.029803, 0.527975],
    [0.200445, 0.017902, 0.593364],
    [0.312543, 0.008239, 0.635700],
    [0.417642, 0.000564, 0.658390],
    [0.517933, 0.021563, 0.654109],
    [0.610667, 0.090204, 0.619951],
    [0.692840, 0.165141, 0.564522],
    [0.764193, 0.240396, 0.502126],
    [0.826588, 0.315714, 0.441316],
    [0.881443, 0.392529, 0.383229],
    [0.928329, 0.472975, 0.326067],
    [0.965024, 0.559118, 0.268513],
    [0.988260, 0.652325, 0.211364],
    [0.994141, 0.753137, 0.161404],
    [0.977995, 0.861432, 0.142808],
    [0.940015, 0.975158, 0.131326]
]


# Diverging colormaps creater with color brewer (http://colorbrewer2.org)
_BrBG =     [(0.6510, 0.3804, 0.1020), (0.8745, 0.7608, 0.4902), (0.9608, 0.9608, 0.9608),
             (0.5020, 0.8039, 0.7569), (0.0039, 0.5216, 0.4431)]
_PiYG =     [(0.8157, 0.1098, 0.5451), (0.9451, 0.7137, 0.8549), (0.9686, 0.9686, 0.9686),
             (0.7216, 0.8824, 0.5255), (0.3020, 0.6745, 0.1490)]
_PRGn =     [(0.4824, 0.1961, 0.5804), (0.7608, 0.6471, 0.8118), (0.9686, 0.9686, 0.9686),
             (0.6510, 0.8588, 0.6275), (0.0000, 0.5333, 0.2157)]
_PuOr =     [(0.9020, 0.3804, 0.0039), (0.9922, 0.7216, 0.3882), (0.9686, 0.9686, 0.9686),
             (0.6980, 0.6706, 0.8235), (0.3686, 0.2353, 0.6000)]
_RdBu =     [(0.7922, 0.0000, 0.1255), (0.9569, 0.6471, 0.5098), (0.9686, 0.9686, 0.9686),
             (0.5725, 0.7725, 0.8706), (0.0196, 0.4431, 0.6902)]
_RdGy =     [(0.7922, 0.0000, 0.1255), (0.9569, 0.6471, 0.5098), (1.0000, 1.0000, 1.0000),
             (0.7294, 0.7294, 0.7294), (0.2510, 0.2510, 0.2510)]
_RdYlBu =   [(0.8431, 0.0980, 0.1098), (0.9922, 0.6824, 0.3804), (1.0000, 1.0000, 0.7490),
             (0.6706, 0.8510, 0.9137), (0.1725, 0.4824, 0.7137)]
_RdYlGn =   [(0.8431, 0.0980, 0.1098), (0.9922, 0.6824, 0.3804), (1.0000, 1.0000, 0.7490),
             (0.6510, 0.8510, 0.4157), (0.1020, 0.5882, 0.2549)]
_Spectral = [(0.8431, 0.0980, 0.1098), (0.9922, 0.6824, 0.3804), (1.0000, 1.0000, 0.7490),
             (0.6706, 0.8667, 0.6431), (0.1686, 0.5137, 0.7294)]


COLORMAPS ={
    # Sequential and perceptually uniform
    "magma": torch.tensor(_magma),
    "plasma": torch.tensor(_plasma),
    "inferno": torch.tensor(_inferno),
    "viridis": torch.tensor(_viridis),
    # Diverging
    "BrBG": torch.tensor(_BrBG),
    "PiYG": torch.tensor(_PiYG),
    "PRGn": torch.tensor(_PRGn),
    "PuOr": torch.tensor(_PuOr),
    "RdBu": torch.tensor(_RdBu),
    "RdGy": torch.tensor(_RdGy),
    "RdYlBu": torch.tensor(_RdYlBu),
    "RdYlGn": torch.tensor(_RdYlGn),
    "Spectral": torch.tensor(_Spectral)
}


def _demo():
    import matplotlib.pyplot as plt
    import numpy as np
    H, W = 32, 512
    scale = torch.tensor(np.tile(np.linspace(0, 1, W), [H, 1]))
    sep = 0.5 * np.ones([H // 8, W, 3])
    images = [sep]
    for k in sorted(COLORMAPS):
        print(k)
        images.append(apply_map(scale, dim=-1, cmap=k).numpy())
        images.append(sep)
    plt.imshow(np.vstack(images))
    plt.show()


if __name__ == "__main__":
    _demo()
