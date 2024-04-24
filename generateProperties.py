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

def loadData(iCount, onnxFile, dataset, data_dir: str = "./tmp"):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()

    data = None
    if dataset == 'MNIST':
        data = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=trns_norm)
    elif dataset == 'CIFAR':
        data = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)
    else:
        raise RuntimeError("Dataset provided is not implemented.")

    loader_test = DataLoader(data, batch_size=10000)

    images, labels = next(iter(loader_test))

    num_selected = 0
    selected_images, selected_labels = [], []

    sess = onnxrun.InferenceSession(onnxFile)

    i = -1
    while num_selected < iCount:
        i += 1
        correctly_classified = True
        input_name = sess.get_inputs()[0].name
        # TODO: Change how input is reshaped depending on benchmark
        if dataset == 'MNIST':
            if 'eran' in onnxFile:
                result = np.argmax(sess.run(None, {input_name: images[i].unsqueeze(0).numpy().astype(np.float32)})[0])
            else:
                result = np.argmax(sess.run(None, {input_name: images[i].numpy().reshape(1, 784, 1)})[0])
        elif dataset == 'CIFAR':
            result = np.argmax(sess.run(None, {input_name: images[i].numpy().reshape(1, 784, 1)})[0])
        if result != labels[i]:
            continue
        num_selected += 1
        selected_images.append(images[i])
        selected_labels.append(labels[i])

    return selected_images, selected_labels

def perturbInstance(instance, eps, mean, std):
    bounds = torch.zeros((*instance.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clip((instance - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((instance + eps), 0, 1) - mean) / std
    return bounds.view(-1, 2)

def saveVnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):
    with open(spec_path, "w") as f:
        f.write(f"; Property with label: {label}.\n")

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
        props += [f"prop_{i}_{eps:.8f}.vnnlib" for i in range(iCount)]
    with open(csvPath, "w") as f:
        for prop in props:
            if prop == props[-1]:
                f.write(f"{onnxModel},{prop},{timeout}")
            else:
                f.write(f"{onnxModel},{prop},{timeout}\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ic", "--instanceCount", type=int, default=50, help="Number of instances to generate properties for")
    ap.add_argument("-o", "--onnxFile", type=str, required=True, help="Path to model onnx file")
    ap.add_argument("-s", "--specFile", type=str, default="props/instances.csv", help="Path to CSV file to store the instance specifications")
    ap.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to use to generate properties")
    ap.add_argument("-ec", "--epsilonCount", type=int, default=10, help="Number of epsilon to sweep over")
    ap.add_argument("-se", "--startEpsilon", type=float, default=0.0, help="Starting epsilon")
    ap.add_argument("-ee", "--endEpsilon", type=float, default=0.04, help="Ending epsilon")
    args = ap.parse_args()

    # 0 -> 0.01: Sweeping epsilon
    #epss = []
    #for i in range(args.epsilonCount):
    #    epss.append(i/255 + .01/255)
    epss = np.linspace(args.startEpsilon, args.endEpsilon, args.epsilonCount) + (.01/255)
    instanceCount = args.instanceCount

    onnxFile = args.onnxFile
    specFile = args.specFile
    dataset = args.dataset.upper()
    imgMean = 0
    imgStd = 1
    if dataset == 'CIFAR':
        imgMean = torch.tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
        imgStd = torch.tensor((0.2471, 0.2435, 0.2616)).view(-1, 1, 1)

    images, labels = loadData(iCount=instanceCount, onnxFile=onnxFile, dataset=dataset)
    for eps in epss:
        for i in range(instanceCount):
            image, label = images[i], labels[i]
            inputBounds = perturbInstance(image, eps, imgMean, imgStd)

            specPath = f"props/{dataset.lower()}/prop_{i}_{eps:.8f}.vnnlib"
            saveVnnlib(inputBounds, label, specPath)
    createInstanceCSV(instanceCount, epss, onnxFile[onnxFile.rfind('/')+1:], specFile, 240)

