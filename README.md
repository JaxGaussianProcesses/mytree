# My🌳

"**M**odule p**ytree**s" that cleanly handle parameter **trainability** and **transformations** for JAX models.

## Installation
```bash
pip install mytree
```

## Usage

### Defining a model 
- Define all your class attributes upfront as an annotation (a bit like a dataclass!). 
- Mark 🍀 attributes with `param_field` to set a default bijector and trainable status.
- Unmarked 🍀 attributes behave as `param_field(bijector=Identity trainable=True)`.

```python
from mytree import Mytree, param, Softplus, Identity

class SimpleModel(Mytree):
    # Marked ☘️ to set default bijector and trainability.
    weight: float = param_field(bijector=Softplus, trainable=False)
    
    # Unmarked ☘️ has `Identity` bijector and trainability set to `True`.
    bias: float 

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def __call__(self, test_point):
        return test_point * self.weight + self.bias
```
- We are fully compatible with [Distrax](https://github.com/deepmind/distrax) and [TensorFlow Probability](https://www.tensorflow.org/probability) bijectors, so feel free to use these!
- As `Mytree` inherits from [simple-pytree's](https://github.com/cgarciae/simple-pytree) `Pytree`, you can mark fields as static via `simple_pytree.static_field`.

### Dataclasses
You can seamlessly use the `dataclasses.dataclass` decorator with `Mytree` classes.

```python
from dataclasses import dataclass

@dataclass
class SimpleModel(Mytree):
    weight: float = param_field(bijector=Softplus, trainable=False)
    bias: float
    
    def __call__(self, test_point):
        return test_point * self.weight + self.bias
```

### Replacing values
Update values via `replace`.

```python
model = SimpleModel(1.0, 2.0)
model.replace(weight=123.0)
```

```
SimpleModel(weight=123.0, bias=2.0)
```
 
## Transformations 🤖

### Applying transformations
Use `constrain` / `unconstrain` to return a `Mytree` with each parameter's bijector `forward` / `inverse` operation applied!
    
```python
model.constrain()
model.unconstrain()
```
    
```
SimpleModel(weight=1.3132616, bias=2.0)
SimpleModel(weight=0.5413248, bias=2.0)
```

### Replacing transformations
Default transformations can be replaced on an instance via the `replace_bijector` method.
```python
new = model.replace_bijector(bias=Identity)
```
```python
new.constrain()
new.unconstrain()
```

```
SimpleModel(weight=1.0, bias=2.0)
SimpleModel(weight=1.0, bias=2.0)
```

And we see that `weight`'s gradient is no longer zero.

## Trainability 🚂

### Applying trainability

We begin by creating some simulated data.
```python
import jax

n = 100
key = jax.random.PRNGKey(123)
x = jax.random.uniform(key, (n, ))
y = 3.0 * x + 2.0 + 1e-3 * jax.random.normal(key, (n, ))
```
And create a mean-squared-error loss function to evaluate our model on.
```python
def loss(model: SimpleModel) -> float:
   model = model.stop_gradient()
   return jax.numpy.sum((y - model(x))**2)
```
Here we use the `stop_gradient` method **within** the loss function, to prevent the flow of gradients during forward or reverse-mode automatic differentiation.
```python
jax.grad(loss)(model)
```
```
SimpleModel(weight=0.0, bias=-188.37418)
```
As `weight` trainability was set to `False`, it's gradient is zero as expected!
    
### Replacing trainability
Default trainability status can be replaced on an instance via the `replace_trainable` method.
```python
new = model.replace_trainable(weight=True)
jax.grad(loss)(model)
```
```
SimpleModel(weight=-121.42676, bias=-188.37418)
```
And we see that `weight`'s gradient is no longer zero.

## Performance 🏎

This is an early experimental library to demonstrate an idea, so it is not yet optimised. Some benchmarks can be found in: https://github.com/Daniel-Dodd/mytree/blob/master/benchmarks/benchmarks.ipynb
