import torch


def compute_estimate(rgb, weights, noise=0.0):
    # rgb =  torch.nn.functional.normalize(rgb)  # !!!
    x = torch.sum(torch.sum(rgb * weights, 3), 2)
    x = x + noise * torch.randn_like(x)
    return torch.nn.functional.normalize(x)


def __compute_estimate(rgb, weights):
    # Basic version
    x = torch.sum(torch.sum(rgb * weights, 3), 2)
    return torch.nn.functional.normalize(x)


def __compute_estimate(rgb, weights):
    # From gradients
    g = spatial_gradient(rgb)
    m = gradient_magnitude(g)
    x = torch.sum(torch.sum(m * weights, 3), 2)
    return torch.nn.functional.normalize(x)


def cosine_loss(estimate, target, reduce=True):
    # Assume all vectors already have unit norm
    cosines = torch.sum(estimate * target, 1)
    loss = 1 - cosines
    angular_error = 180 * torch.acos(torch.clamp(cosines, -1, 1)) / 3.141592653589793
    if reduce:
        loss = torch.mean(loss)
        angular_error = torch.mean(angular_error)
    return loss, angular_error


def apply_correction(rgb, estimates):
    norm = torch.prod(estimates, 1, keepdim=True) ** (1.0 / 3)
    out = norm[:, :, None, None] * rgb / (1e-6 + estimates[:, :, None, None])
    return torch.clamp(out, 0, 1)


def scaled_intensity(rgb):
    y = torch.mean(rgb, 1, keepdim=True)
    ymin = torch.min(y.view(y.size(0), -1), 1)[0]
    y = y - ymin[:, None, None, None]
    ymax = torch.max(y.view(y.size(0), -1), 1)[0]
    y = y / (1e-7 + ymax)[:, None, None, None]
    return y


_sobel_mask = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_sobel_filters = torch.stack([_sobel_mask, _sobel_mask.t()] * 3, 0).view(-1, 1, 3, 3)


def spatial_gradients(x):
    # input x : BxCxHxW
    # output y: Bx(2C)xHxW
    # the horiz. and vert. components of the gradient of x[b, c, i, j]
    # are out[b, 2*c, i, j] and out[b, 2*c+1, i, j]
    w = _sobel_filters.to(x.device)
    y = torch.nn.functional.conv2d(x, w, padding=1, groups=x.size(1))
    return y


def gradient_magnitude(g):
    # input g  : Bx(2C)xHxW
    # output m : BxCxHxW
    n, _, h, w = g.size()
    return torch.norm(g.view(n, -1, 2, h, w), 2, 2)


def normalize_gradients(g):
    # input g  : Bx(2C)xHxW
    # output m : Bx(2C)xHxW
    n, _, h, w = g.size()
    nrm = torch.nn.functional.normalize(g.view(n, -1, 2, h, w), 2, 2)
    return nrm.view(n, -1, h, w)


def scaled_intensity_and_gradient_directions(rgb):
    # y = scaled_intensity(rgb)
    g = normalize_gradients(spatial_gradients(rgb))
    return torch.cat([y, g], 1)


def _equalize1(image, vmin, vmax, bins):
    scaled = (image - vmin) * (bins - 2) / (vmax - vmin)
    indices = scaled.long()
    hist = torch.bincount(indices.view(-1), minlength=bins)
    chist = torch.cumsum(hist, 0).float()
    chist -= chist[0]
    chist /= chist[-1].clamp(min=1)
    f = scaled - indices.float()
    interpolated = (1 - f) * chist[indices] + f * chist[indices + 1]
    return interpolated
    

def histogram_equalization(data, dim=1, vmin=0, vmax=1, bins=256):
    """Histogram equalization.
    
    Normalize elements in data.  The normalization is performed
    indipendently from elements starting from dimension `dim'
    (e.g. dim=1 normalizes independently each element in a batch).

    vmin, vmax and bins defin the quantization used for the histogram
    computation.

    """
    sz = 1
    for i in range(dim):
        sz *= data.size(i)
    vdata = data.view(sz, -1)
    output = torch.empty_like(vdata)
    for i in range(sz):
        output[i] = _equalize1(vdata[i], vmin, vmax, bins)
    return output.view(data.size())
