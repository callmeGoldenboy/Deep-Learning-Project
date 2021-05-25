# Experiments on the AlexNet model
Project in the course DD2424 Deep Learning in Data Science, which consists of a set of experiments on the AlexNet model. The conducted experiments were:
* Investigate the effects of batch normalization
* Investigate the effects of dropout
* Investigate the effects of data-augmentation

## Environment

The project is implemented with the keras API which uses the Tensorflow 2.5 plattform.

Prepare a virtual environment with python>=3.9, and install the dependencies with

```bash
pip install -r requirements.txt
```

## Data 

The  experiments were conducted on the [ciphar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

Instantiating a model can be done with the init_model function. One can set the settings of the network, such as having batch normalization or not, having dropout etc. 

```python
def init_model(params):
    """Initializes an AlexNet model with the given parameter settings

    Args:
        params (dict, optional): the settings to apply to the model. Defaults to {}.

    Returns:
        AlexNet object: the initialized AlexNet object
    """
    new_params = {
        "batch_norm":False,
        "data_augmentation": False,
        "dropout": False,
        "annealer": False,
        "batch_size": 100,
        "epoch":5,
        "learn_rate":.001
    } 
    for k,v in params.items():
        new_params[k] = v 

    print("Model initated w. params:", new_params)
    model = AlexNet(new_params)

    return model

```
