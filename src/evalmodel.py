#!/usr/bin/env python3

import torch
import argparse
import os
import model
import data
import processing
import numpy as np
import pickle
import scipy.misc
import ptcolor
import ptcolormap


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the achromatic pixel detector")
    a = parser.add_argument
    a("model_file", help="Model to evaluate")
    a("test_list", help="File listing test images")
    a("--image-size", type=int, default=256, help="Size of input images")
    a("--remove-gamma", action="store_true", help="Remove srgb gamma from training images.")
    a("--apply-gamma", action="store_true", help="Apply srgb gamma to output data.")
    a("--mask-clipped", action="store_true", help="Exclude clipped pixels from the estimate")
    a("--mask-black", action="store_true", help="Exclude black pixels from the estimate")
    a("--batch-size", type=int, default=16, help="Size of the minibatch")
    a("--num-workers", type=int, default=torch.get_num_threads(), help="Number of parallel threads")
    a("--device", default="cuda", help="Processing device")
    a("--plot-estimates", action="store_true", help="show the estimates on a plot")
    a("--filter-outliers", action="store_true", help="exclude pixels outside the range of allowed illuminants (deprecated)")
    a("--filter", help="Classifier excluding unlikely rgb values.")
    a("--cv", type=int, help="Number of cross validation folds")
    a("--tex", action="store_true", help="Latex table format")
    a("--gw", action="store_true", help="Apply gray-world instead")
    a("--gt", action="store_true", help="Use the ground truth instead of the actual estimate")
    a("--output-dir", help="Directory where processed images are placed")
    return parser.parse_args()


def median(seq):
    l = sorted(seq)
    n = len(l)
    if n == 0:
        return float("nan")
    elif n % 2 == 1:
        return l[(n - 1) // 2]
    else:
        return 0.5 * (l[n // 2] + l[(n - 2) // 2])


def plot_estimates(estimates, illuminants):
    import matplotlib.pyplot as plt
    diff = estimates - illuminants
    for i in range(3):
        j = (i + 1) % 3
        plt.subplot(2, 2, i + 1)
        plt.grid()
        plt.quiver(illuminants[:, i], illuminants[:, j], diff[:, i], diff[:, j], color=estimates)
        plt.xlabel("RGB"[i]);
        plt.ylabel("RGB"[j]);
    plt.show()


def filter_outliers(rgb, w):
    vmin = [0.17362045760430686, 0.3695501042804648, 0.10246227164416204]
    vmax = [0.517077045274027, 0.512396694214876, 0.4411560234382759]
    rgb_n = torch.nn.functional.normalize(rgb, 1)
    mask_min = torch.min((rgb_n >= rgb_n.new_tensor(vmin)[None, :, None, None]), 1, keepdim=True)[0]
    mask_max = torch.min((rgb_n <= rgb_n.new_tensor(vmax)[None, :, None, None]), 1, keepdim=True)[0]
    mask = mask_min.float() * mask_max.float()
    return w * mask


def apply_filter(filter_, rgb, w):
    if filter_ is None:
        return w
    rg = torch.nn.functional.normalize(rgb, 1)[:, :2, ...].cpu().detach().numpy()
    rg_s = rg.transpose([0, 2, 3, 1])
    z = filter_.predict(rg_s.reshape(-1, 2))
    sh = [rg_s.shape[0], 1, rg_s.shape[1], rg_s.shape[2]]
    z = z.reshape(sh)
    return w * (w.new_tensor(z) > 0).float()


def process_test_set(args, fold=None):
    model_data = torch.load(args.model_file.format(fold=fold))
    if not args.tex:
        print("Training args:", model_data["args"])
    input_code = model_data["args"].input
    output_code = model_data["args"].output

    net = model.CCNet(input_code=input_code, output_code=output_code, noise=0.0, mask_clipped=args.mask_clipped)
    net.load_state_dict(model_data["model"])
    net.to(args.device)
    net.eval()
    dataset = data.AchromaticDataset(args.test_list.format(fold=fold), False,
                                     args.image_size, include_path=True, include_fullres=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    angular_errors = []
    estimates = []
    targets = []
    if args.filter is not None:
        with open(args.filter.format(fold=fold), "rb") as f:
            filter_ = pickle.load(f)
    else:
            filter_ = None

    print_ = (lambda *a,**kw: None if args.tex else print)
            
    for rgb, illuminant, path, fullres in loader:
        rgb = rgb.to(args.device)
        if args.remove_gamma:
            fullres_ng = ptcolor.remove_gamma(fullres)
            rgb = ptcolor.remove_gamma(rgb)
        else:
            fullres_ng = fullres
        illuminant = illuminant.to(args.device)
        illuminant = torch.nn.functional.normalize(illuminant)
        with torch.no_grad():
            if args.gw:
                weights = rgb.new_ones(rgb.size(0), 1, rgb.size(2), rgb.size(3))
            else:
                _, _, weights = net(rgb)
            if args.mask_black:
                weights = weights * (rgb.min(1, keepdim=True)[0] > 0).float()
            if args.filter_outliers:
                weights = filter_outliers(rgb, weights)
            weights = apply_filter(filter_, rgb, weights)
            if not args.gt:
                estimate = net.make_estimate(rgb, weights)
            else:
                estimate = illuminant
        loss, ang_err = processing.cosine_loss(estimate, illuminant, reduce=False)
        print_(".", end="", flush=True)
        angular_errors.extend(ang_err.cpu().numpy())
        targets.append(illuminant.cpu().numpy())
        estimates.append(estimate.cpu().numpy())
        if args.output_dir is not None:
            balanced = processing.apply_correction(fullres_ng, estimate.cpu())
            if args.apply_gamma:
                if not args.remove_gamma:
                    fullres = ptcolor.apply_gamma(fullres_ng)
                balanced = ptcolor.apply_gamma(balanced)
                estimate = ptcolor.apply_gamma(estimate.unsqueeze(-1).unsqueeze(-1)).view(-1, 3)
                illuminant = ptcolor.apply_gamma(illuminant.unsqueeze(-1).unsqueeze(-1)).view(-1, 3)
            w = ptcolormap.apply_map(weights, 0, 1).squeeze(2)
            for k in range(balanced.size(0)):
                name = os.path.splitext(os.path.basename(path[k]))[0]
                im = fullres[k].numpy().transpose([1, 2, 0])
                im = (im * 255).astype(np.uint8)
                p = os.path.join(args.output_dir, name + "_input.png")
                scipy.misc.imsave(p, im)
                im = balanced[k].cpu().numpy().transpose([1, 2, 0])
                im = (im * 255).astype(np.uint8)
                p = os.path.join(args.output_dir, name + "_balanced.png")
                scipy.misc.imsave(p, im)
                p = os.path.join(args.output_dir, name + "_weights.png")
                im = w[k].cpu().numpy().transpose([1, 2, 0])
                im = (im * 255).astype(np.uint8)
                scipy.misc.imsave(p, im)
                p = os.path.join(args.output_dir, "estimate.txt")
                with open(p, "a") as f:
                    print(os.path.basename(path[k]),
                          *estimate[k].cpu().numpy(),
                          *illuminant[k].cpu().numpy(),
                          ang_err[k].item(), file=f)
                
    print_()
    return angular_errors, estimates, targets


def main():
    args = parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    if args.cv is None:
        angular_errors, estimates, targets = process_test_set(args, None)
    else:
        angular_errors, estimates, targets = [], [], []
        for fold in range(1, args.cv + 1):
            a, e ,t = process_test_set(args, fold)
            angular_errors.extend(a)
            estimates.extend(e)
            targets.extend(t)
            
    mean_err = sum(angular_errors) / len(angular_errors)
    median_err = median(angular_errors)
    max_err = max(angular_errors)
    if args.tex:
        fmt = "{:10}s & {:.2f} & {:.2f} & {:.2f} \\\\"
        print(fmt.format(args.model_file, mean_err, median_err, max_err))
    else:
        print(len(angular_errors), "test images")
        print("MEAN   MEDIAN MAX (degrees)")
        print("{:.4f} {:.4f} {:.4f}".format(mean_err, median_err, max_err))
    estimates = np.vstack(estimates)
    targets = np.vstack(targets)
    if args.plot_estimates:
        plot_estimates(estimates, targets)
        

if __name__ == "__main__":
    main()
    
