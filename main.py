import torch.onnx
import torch

def storeModelAsOnnx(fModel, fOnnx, inputSize):
    model = torch.load(fModel)
    torch.onnx.export(model, inputSize, fOnnx)

def getModelFromOnnx(fOnnx):
    return torch.onnx.load(fOnnx)

def getPthAsOnnx(fModel, fOnnx, inputSize):
    storeModelAsOnnx(fModel, fOnnx, inputSize)
    return getModelFromOnnx(fOnnx)