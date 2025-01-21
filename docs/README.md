# Train the Pong game model

## Step 1. install the conda and create the environment

```shell
conda create --name reinforcement python=3.10
```
when the conda env created, install jupyter notebook in the env
```shell
pip install jupyter notebook
```

## Step 2. install the python packages

```shell
pip install swig
pip install box2d-py
```
 - SWIG is a software development tool that connects programs written in C and C++ with a variety of high-level programming languages
 - box2d-py is a 2D physics library in Python

the environment needs the above packages to simulate the physical world.

## Step 3. train the model

execute the following command to start training with episode 50000 and learning rate 0.001

```shell
python train_reinforce.py --episode 50000 --lr 0.001
```
when the training stops, two `.pth` files will be saved in the same directory.
 - `reinforce_model.pth` the model layers and its weights in PyTorch format
 - `reinforce_model_params.pth`  the model weights without model layers in PyTorch format

## Step 4. verify the model

in the conda env `reinforcement`, start the jupyter notebook service
```shell
jupyter notebook
```
open the `train_test.ipynb` in the jupyter, verify the model

## Step 5. convert the `.pth` model to `.onnx` model

execute the follow command to convert the `reinforce_model.pth` to `reinforce_model.onnx`
```shell
python convert_pytorch_to_onnx.py
```
then the `reinforce_model.onnx` is generated, copy it to the folder `resources/weight/`

