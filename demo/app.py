#!/usr/bin/env python3

import flask
from PIL import Image
import numpy as np
import os
import time
import random
import urllib.request
import qucc


IMSIZE = 512

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/images"
app.config['MAX_CONTENT_LENGTH'] = 3 * (1024 ** 2)


qucc_model = qucc.QUCC()
print("Model initialized")


def error(message):
    return flask.render_template("demo.html", error=message)


def make_name():
    num = hash(time.time()) % 100000
    return f"{num:05d}"


def process(im, srgb):
    im = im / 255.0
    inimg, out, w, estimate = qucc_model.process(im, srgb)
    inimg = np.clip(inimg * 255, 0, 255).astype(np.uint8)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    w = np.clip(w * 255, 0, 255).astype(np.uint8)
    estimate = np.clip(estimate * 255, 0, 255).astype(np.uint8)
    return inimg, out, w, estimate


def process_image(im, srgb):
    if max(im.height, im.width) > 1536:
        return error("The image is too big!")
    elif min(im.height, im.width) < 64:
        return error("The image is too small!")
    filename = make_name()
    sz = max(im.width, im.height)
    newh = (im.height * IMSIZE) // sz
    neww = (im.width * IMSIZE) // sz
    im = im.resize((neww, newh))
    inimg, out, weights, estimate = process(np.array(im), srgb)
    folder = app.config['UPLOAD_FOLDER']
    file_in = os.path.join(folder, filename + "-in.jpg")
    inimg = Image.fromarray(inimg)
    inimg.save(file_in)
    out = Image.fromarray(out)
    file_out = os.path.join(folder, filename + "-out.jpg")
    out.save(file_out)
    weights = Image.fromarray(weights)
    file_weights = os.path.join(folder, filename + "-weights.jpg")
    weights = weights.resize((neww, newh))
    weights.save(file_weights)
    estimate = Image.fromarray(estimate[None, None, :])
    file_estimate = os.path.join(folder, filename + "-estimate.jpg")
    estimate = estimate.resize((neww, newh))
    estimate.save(file_estimate)
    return flask.render_template("demo.html", file_in=file_in, file_out=file_out,
                                 file_weights=file_weights, file_estimate=file_estimate)


@app.route("/demo", methods=("GET",))
def demo():
    return flask.render_template("demo.html")


@app.route("/upload", methods=("POST",))
def upload():
    srgb = (flask.request.form.get("srgb") == "srgb")

    # 1) Download the image from an url (if any) and process it
    url = flask.request.form.get("imageurl")
    if url:
        headers={"User-Agent": "Wget/1.19.4 (linux-gnu)", "Accept": "*/*"}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as f:
                im = Image.open(f)
        except Exception as e:
            return error(f"Error processing the URL ({e})")
        return process_image(im, srgb)

    # 2) Process the uploaded image (if any)
    upfile = flask.request.files.get("upload")
    if upfile and upfile.filename != "":
        try:
            im = Image.open(upfile)
        except Exception as e:
            return error(f"Error decoding the image ({e})")
        return process_image(im, srgb)
        
    # 3) Report an error    
    return error("Select an image or an URL")


@app.route("/random", methods=("GET",))
def random_examples():
    folder = app.config['UPLOAD_FOLDER']
    outs = os.listdir(folder)
    outs = [os.path.join(folder, o) for o in outs if o.endswith("-out.jpg")]
    outs = random.sample(outs, 10)
    ins = [o.replace("-out.jpg", "-in.jpg") for o in outs]
    return flask.render_template("random.html", n=len(ins), ins=ins, outs=outs)
