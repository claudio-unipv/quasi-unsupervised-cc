#!/usr/bin/env python3

import torch
import torchvision
import tensorboardX
import argparse
import os
import signal
import collections
import time
import model
import data
import processing
import ptutils
import ptcolormap
import ptcolor


def parse_args():
    DEFAULT_TRAINING_LIST = "train.txt"
    parser = argparse.ArgumentParser(description="Train the achromatic pixel detector")
    a = parser.add_argument
    a("output_dir", help="Directory where training data is placed")
    a("--training-list", default=DEFAULT_TRAINING_LIST, help="File listing training images")
    a("--image-size", type=int, default=256, help="Size of input images")
    a("--iterations", type=int, default=350000, help="Number of training steps")
    a("--save-every", type=int, default=5000, help="Frequency at which models are saved")
    a("--checkpoint-every", type=int, help="Frequency at which checkpoints are saved")
    a("--start-from", help="Start from the given pretrained model")
    a("--validation-list",  help="File listing validation images")
    a("--validate-every", type=int, default=1000, help="Frequency at which models are validated")
    a("--batch-size", type=int, default=16, help="Size of the minibatch")
    a("--learning-rate", type=float, default=1e-4, help="Learning rate")
    a("--lambda", type=float, dest="lambda_", default=0.0, help="Coefficient for the weight term")
    a("--input", default="eg", help="Definition of the input of the network")
    a("--output", default="c", help="Definition of the data to use for the estimate")
    a("--last-only", action="store_true", help="Train only the last layer")
    a("--weight-decay", type=float, default=1e-5, help="Weight decay")
    a("--noise", type=float, default=100, help="Amount of noise to introduce in the estimate.")
    a("--remove-gamma", action="store_true", help="Remove srgb gamma from training images.")
    a("--mask-clipped", action="store_true", help="Exclude clipped pixels from the estimate")
    a("--validation-remove-gamma", action="store_true", help="Remove srgb gamma from validation images.")
    a("--num-workers", type=int, default=torch.get_num_threads(), help="Number of parallel threads")
    a("--device", default="cuda", help="Processing device")
    return parser.parse_args()


def cycle(seq):
    while True:
        for x in seq:
            yield x


def make_summary_image(fmt, *images):
    # C -> color, G -> gray, M -> colormap
    all = []
    for f, im in zip(fmt, images):
        if f == "C":
            all.append(im)
        elif f == "G":
            vmin = im.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
            vmax = im.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
            nrm = (im - vmin) / (vmax - vmin + 1e-6)
            all.append(nrm.repeat(1, 3, 1, 1))
        elif f == "M":
            all.append(ptcolormap.apply_map(im.squeeze(1), 0, 1))
        else:
            all.append(torch.zeros_like(im))
    disp = torch.cat(all, 3)
    return torchvision.utils.make_grid(disp, nrow=2, padding=1)


def save_model(path, model, optimizer, step, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": args,
    }, path + ".temp")
    os.rename(path + ".temp", path)  # Replace atomically


def initialization(path, model, optimizer):
    try:
        data = torch.load(path)
    except FileNotFoundError:
        print("Starting from step 1");
        return 0
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    print("Continue from step", data["step"])
    return data["step"]


def validate_model(model, args):
    dataset = data.AchromaticDataset(args.validation_list, False, args.image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    model.eval()
    errors = []
    with torch.no_grad():
        for rgb_cpu, illuminant in loader:
            rgb = rgb_cpu.to(args.device)
            if args.validation_remove_gamma:
                rgb = ptcolor.remove_gamma(rgb)
            illuminant = illuminant.to(args.device)
            illuminant = torch.nn.functional.normalize(illuminant)
            estimate = model(rgb)[0]
            loss, ang_err = processing.cosine_loss(estimate, illuminant)
            errors.append(ang_err.item())
    model.train()
    return sum(errors) / max(1, len(errors))


def select_parameters(net, last_only):
    params = net.parameters()
    if last_only:
        for p in params:
            p.requires_grad = False
        params = list(net.last_parameters())
        for p in params:
            p.requires_grad = True
    return params


def main():
    args = parse_args()
    net = model.CCNet(noise=args.noise, input_code=args.input, output_code=args.output, mask_clipped=args.mask_clipped)
    net.to(args.device)

    optimizer = torch.optim.Adam(select_parameters(net, args.last_only),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    dataset = data.AchromaticDataset(args.training_list, True, args.image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
    model_path = os.path.join(args.output_dir, "model.pt")
    step = initialization(model_path, net, optimizer)
    if args.start_from is not None:
        net.load_state_dict(torch.load(args.start_from)["model"])

    writer = tensorboardX.SummaryWriter(args.output_dir)
    # writer.add_text("Options", str(args), step)
    display = ptutils.Display("   ".join(["Step {step}", "loss {loss:.5f}",
                                          "angular {ang_err:.3f} degs",
                                          "speed {steps_s:.2f} steps/s"]))
    display_validation = ptutils.Display("STEP {step}   VALIDATION ERROR {valid_err:.4f} DEGS")
    train_loss_history = collections.deque(maxlen=100)
    train_ang_history = collections.deque(maxlen=100)

    interrupted = False
    def handler(sig, frame):
        nonlocal interrupted
        interrupted = interrupted or print("Training interrupted") or True
    signal.signal(signal.SIGINT, handler)

    data_iter = cycle(loader)
    for rgb_cpu, illuminant in data_iter:
        if step >= args.iterations or interrupted:
            break
        rgb = rgb_cpu.to(args.device)
        if args.remove_gamma:
            rgb = ptcolor.remove_gamma(rgb)
        illuminant = illuminant.to(args.device)
        illuminant = torch.nn.functional.normalize(illuminant)
        optimizer.zero_grad()
        estimate, preprocessed, weights = net(rgb)
        loss, ang_err = processing.cosine_loss(estimate, illuminant)
        if args.lambda_ > 0:
            loss = loss - args.lambda_ * weights.mean()
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())
        train_ang_history.append(ang_err.item())
        step += 1
        if step % 100 == 0:
            mean_loss = sum(train_loss_history) / max(1, len(train_loss_history))
            mean_ang_err = sum(train_ang_history) / max(1, len(train_ang_history))
            writer.add_scalar("loss", mean_loss, step)
            writer.add_scalar("angular_error", mean_ang_err, step)
            display.disp(step, loss=mean_loss, ang_err=mean_ang_err)
        if step % args.validate_every == 0 and args.validation_list is not None:
            valid_err = validate_model(net, args)
            writer.add_scalar("validation_error", valid_err, step)
            display_validation.disp(step, valid_err=valid_err)
        if step % 1000 == 0:
            estimate = net.make_estimate(rgb, weights, noise=0.0)
            balanced = processing.apply_correction(rgb, estimate)
            balanced[:, :, 16:48, 16:32] = illuminant[:, :, None, None]
            balanced[:, :, 16:48, 32:48] = estimate[:, :, None, None]
            try:
                sum_image = make_summary_image("GMCC", preprocessed[:, 0:1, :, :].cpu(),
                                               weights.mean(1, keepdim=True).cpu(),
                                               balanced.cpu(), rgb.cpu())
                writer.add_image("display", sum_image, step)
            except Exception as e:
                print("Warning", e)
        if args.checkpoint_every is not None and step % args.checkpoint_every == 0:
            path = model_path.replace(".pt", "-{}.pt".format(step))
            save_model(path, net, optimizer, step, args)
            print("CHECKPOINT SAVED TO '{}'".format(path))
        if step % args.save_every == 0 or step == args.iterations or interrupted:
            save_model(model_path, net, optimizer, step, args)
            writer.add_text("Save", "Model saved at step {}\n".format(step), step)
    data_iter.close()
    print("Training completed")


if __name__ == "__main__":
    main()
