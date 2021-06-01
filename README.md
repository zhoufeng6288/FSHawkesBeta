# FSHawkesBeta
The repository implements the flexible state-switching Hawkes processes in the paper "Nonlinear Hawkes Processes in Time-Varying System". It includes both simulation and inference.

We recommend to read the tutorial to get familiar with this module. It introduces the key points in the model and illustrates how to perfrom simulation and inference.

For any further details, please refer to the paper.

# Instructions
Please run the command line below for detailed guide:
```
python exp.py --help 
```
## Example 1: synthetic data using Gibbs sampler
```
python exp.py -v 1 -a 1 -ng 200000 -ngt 200000 -niter 200 -d synthetic -m gibbs
```
## Example 2: synthetic data using mean-field variational inference
```
python exp.py -v 1 -a 1 -ng 100 -ngt 100 -niter 200 -d synthetic -m mf
```
## Example 3: SCE data using mean-field variational inference
```
python exp.py -v 1 -a 1 -ng 100 -ngt 100 -niter 200 -d SCE -m mf
```
## Example 4: INTC data using mean-field variational inference
```
python exp.py -v 1 -a 1 -ng 500 -ngt 500 -niter 1000 -d INTC -m mf
```
