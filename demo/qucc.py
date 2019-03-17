import sys
sys.path.append("../src")
import torch
import numpy as np
import model
import ptcolor
import processing
import ptcolormap


MODEL_FILE = "../models/ilsvrc12-eg.pt"
SIZE = 256


class QUCC:
    def __init__(self):
        model_data = torch.load(MODEL_FILE)
        input_code = model_data["args"].input
        output_code = model_data["args"].output
        self.net = model.CCNet(input_code=input_code, output_code=output_code, noise=0.0)
        self.net.load_state_dict(model_data["model"])
        self.net.to("cpu")
        self.net.eval()

    def process(self, fullres, srgb):
        """Process an image

        fullres is in [0,1].
        Return inpute image, processed image and weights all in [0,1].
        """
        fullres = np.transpose(fullres.astype(np.float32), [2, 0, 1])
        fullres = fullres[None, ...]
        fullres = torch.tensor(fullres, dtype=torch.float)
        fullres_ng = (ptcolor.remove_gamma(fullres) if srgb else fullres)
        rgb = torch.nn.functional.interpolate(fullres_ng, (SIZE, SIZE),
                                              mode="bilinear", align_corners=True)
        with torch.no_grad():
            _, _, weights = self.net(rgb)
            weights = weights * (rgb.min(1, keepdim=True)[0] > 0).float()
            estimate = self.net.make_estimate(rgb, weights)
            balanced = processing.apply_correction(fullres_ng, estimate)
            balanced = ptcolor.apply_gamma(balanced)
            balanced = balanced[0].cpu().numpy().transpose([1, 2, 0])
            fullres_g = (fullres if srgb else ptcolor.apply_gamma(fullres))
            fullres_g = fullres_g[0].cpu().numpy().transpose([1, 2, 0])
            estimate = ptcolor.apply_gamma(estimate.unsqueeze(-1).unsqueeze(-1)).view(3)
            estimate = estimate.cpu().numpy()
            wimg = ptcolormap.apply_map(weights, 0, 1).squeeze(2)
            wimg = wimg[0].cpu().numpy().transpose([1, 2, 0])
        return fullres_g, balanced, wimg, estimate
