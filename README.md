# My🌳
[![PyPI version](https://badge.fury.io/py/mytree.svg)](https://badge.fury.io/py/mytree)
[![codecov](https://codecov.io/gh/Daniel-Dodd/mytree/branch/main/graph/badge.svg?token=Q1R280Vb5i)](https://codecov.io/gh/Daniel-Dodd/mytree)


"**M**odule p**ytree**s" that cleanly handle parameter **trainability** and **transformations** for JAX models.

## Installation
```bash
pip install mytree
```

## Usage

### Defining a model
- Mark leaf attributes with `param_field` to set a default bijector and trainable status.
- Unmarked leaf attributes default to an `Identity` bijector and trainablility set to `True`.

```python
from mytree import Mytree, param_field, Softplus, Identity

class SimpleModel(Mytree):
    weight: float = param_field(bijector=Softplus, trainable=False)

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias # Unmarked 🍀 attribute `bias`, has `Identity` bijector and trainability set to `True`.
    
    def __call__(self, test_point):
        return test_point * self.weight + self.bias
```
- Fully compatible with [Distrax](https://github.com/deepmind/distrax) and [TensorFlow Probability](https://www.tensorflow.org/probability) bijectors, so feel free to use these!

### Dataclasses
Works seamlessly with the `dataclasses.dataclass` decorators!

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
And we see that `weight`'s parameter is no longer transformed under the `Identity`.

## Trainability 🚂

### Applying trainability

Applying `stop_gradient` **within** the loss function, prevents the flow of gradients during forward or reverse-mode automatic differentiation.
```python
import jax

# Create simulated data.
n = 100
key = jax.random.PRNGKey(123)
x = jax.random.uniform(key, (n, ))
y = 3.0 * x + 2.0 + 1e-3 * jax.random.normal(key, (n, ))


# Define a mean-squared-error loss.
def loss(model: SimpleModel) -> float:
   model = model.stop_gradient() # 🛑 Stop gradients!
   return jax.numpy.sum((y - model(x))**2)
   
jax.grad(loss)(model)
```
```
SimpleModel(weight=0.0, bias=-188.37418)
```
As `weight` trainability was set to `False`, it's gradient is zero as expected!
    
### Replacing trainability
Default trainability status can be replaced via the `replace_trainable` method.
```python
new = model.replace_trainable(weight=True)
jax.grad(loss)(model)
```
```
SimpleModel(weight=-121.42676, bias=-188.37418)
```
And we see that `weight`'s gradient is no longer zero.

## Metadata

### Viewing `field` metadata
View field metadata pytree via `meta`.
```python
from mytree import meta
meta(model)
```
```
SimpleModel(weight=({'bijector': Bijector(forward=<function <lambda> at 0x17a024e50>, inverse=<function <lambda> at 0x17a024430>), 1.0), 'trainable': False, 'pytree_node': True}, bias=({}, 2.0))
```

Or the metadata pytree leaves via `meta_leaves`.
```python
from mytree import meta_leaves
meta_leaves(model)
```
```
[({}, 2.0),
 ({'bijector': Bijector(forward=<function <lambda> at 0x17a024e50>, inverse=<function <lambda> at 0x17a024430>),
  'trainable': False,
  'pytree_node': True}, 1.0)]
```
Note this shows any metadata defined via a `dataclasses.field` for the pytree leaves. So feel free to define your own.

### Applying `field` metadata
Leaf metadata can be applied via the `meta_map` function.
```python
from mytree import meta_map

# Function passed to `meta_map` has its argument as a `(meta, leaf)` tuple!
def if_trainable_then_10(meta_leaf):
    meta, leaf = meta_leaf
    if meta.get("trainable", True):
        return 10.0
    else:
        return leaf

meta_map(if_trainable_then_10, model)
```
```
SimpleModel(weight=1.0, bias=10.0)
```
It is possible to define your own custom metadata and therefore your own metadata transformations in this vein.

## Static fields
Since `Mytree` inherits from [simple-pytree](https://github.com/cgarciae/simple-pytree)'s `Pytree`, fields can be marked as static via simple_pytree's `static_field`.

```python
import jax.tree_util as jtu
from simple_pytree import static_field

class StaticExample(Mytree):
    b: float = static_field
    
    def __init__(self, a=1.0, b=2.0):
        self.a=a
        self.b=b
    
jtu.tree_leaves(StaticExample())
```
```
[1.0]
```

## Performance 🏎
Preliminary benchmarks can be found in: https://github.com/Daniel-Dodd/mytree/blob/master/benchmarks/benchmarks.ipynb
