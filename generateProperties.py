import torch.onnx
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import numpy as np
import onnxruntime as onnxrun

import argparse
import os
import sys

def storeModelAsOnnx(fModel, fOnnx, inputSize):
    model = torch.load(fModel)
    torch.onnx.export(model, inputSize, fOnnx)

def getModelFromOnnx(fOnnx):
    return torch.onnx.load(fOnnx)

def getPthAsOnnx(fModel, fOnnx, inputSize):
    storeModelAsOnnx(fModel, fOnnx, inputSize)
    return getModelFromOnnx(fOnnx)

def loadData(iCount, onnxFile, data_dir: str = "./tmp"):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    mnist_test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=trns_norm)

    loader_test = DataLoader(mnist_test, batch_size=10000)

    images, labels = next(iter(loader_test))

    num_selected = 0
    selected_images, selected_labels = [], []

    sess = onnxrun.InferenceSession(onnxFile)

    i = -1
    while num_selected < iCount:
        i += 1
        correctly_classified = True
        input_name = sess.get_inputs()[0].name
        result = np.argmax(sess.run(None, {input_name: images[i].numpy().reshape(1, 784, 1)})[0])
        if result != labels[i]:
            correctly_classified = False
            break
        if not correctly_classified:
            continue
        num_selected += 1
        selected_images.append(images[i])
        selected_labels.append(labels[i])

    return selected_images, selected_labels

def perturbInstance(instance, eps):
    bounds = torch.zeros((*instance.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((instance - eps), 0, 1)
    bounds[..., 1] = torch.clip((instance + eps), 0, 1)
    return bounds.view(-1, 2)

def saveVnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):
    with open(spec_path, "w") as f:
        f.write(f"; Mnist property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))")

def createInstanceCSV(iCount, epss, onnxModel, csvPath, timeout=120):
    props = []
    for eps in epss:
        props += [f"props/prop_{i}_{eps:.8f}.vnnlib" for i in range(iCount)]
    with open(csvPath, "w") as f:
        for prop in props:
            if prop == props[-1]:
                f.write(f"{onnxModel},{prop},{timeout}")
            else:
                f.write(f"{onnxModel},{prop},{timeout}\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ic", "--instanceCount", type=int, default=50)
    ap.add_argument("-o", "--onnxFile", type=str, required=True)
    ap.add_argument("-s", "--specFile", type=str)
    ap.add_argument("-ec", "--epsilonCount", type=int, default=10, help="Number of epislon to sweep over")
    args = ap.parse_args()

    # 0 -> 0.01: Sweeping epsilon
    epss = []
    for i in range(args.epsilonCount):
        epss.append((i/5)/255 + .01/255)
    instanceCount = args.instanceCount

    onnxFile = args.onnxFile
    specFile = args.specFile if args.specFile is not None else "props/mnist_instances.csv"

    images, labels = loadData(iCount=instanceCount, onnxFile=onnxFile)
    for eps in epss:
        for i in range(instanceCount):
            image, label = images[i], labels[i]
            inputBounds = perturbInstance(image, eps)

            specPath = f"props/prop_{i}_{eps:.8f}.vnnlib"
            saveVnnlib(inputBounds, label, specPath)
    createInstanceCSV(instanceCount, epss, onnxFile, specFile)

