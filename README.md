# Deep Neural Network Verification Analysis

Setup can be done by following the installation steps for each verifier. Two separate conda environments for each is advised.

AB-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/README.md

NeuralSAT: https://github.com/dynaroars/neuralsat/blob/develop/doc/INSTALL.md


## Property Generation
```python generateProperties.py [-h] [-ic INSTANCECOUNT] -o ONNXFILE [-s SPECFILE] -d DATASET [-ec EPSILONCOUNT]```

optional arguments:

  -h, --help            show this help message and exit

  -ic INSTANCECOUNT, --instanceCount INSTANCECOUNT

                        Number of instances to generate properties for
  
  -o ONNXFILE, --onnxFile ONNXFILE
  
                        Path to model onnx file
  
  -s SPECFILE, --specFile SPECFILE
  
                        Path to CSV file to store the instance specifications
  
  -d DATASET, --dataset DATASET
  
                        Dataset to use to generate properties
  
  -ec EPSILONCOUNT, --epsilonCount EPSILONCOUNT
  
                        Number of epsilon to sweep over


Once the cooresponding .vnnlib files and .csv file is made, these can be given to the verifier

