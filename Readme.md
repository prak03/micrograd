# micrograd, rebuilt from scratch

My own rebuild of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd): a tiny autograd engine and a small neural network library, written in plain Python on single numbers.

I built it with one rule: no copy-pasting, and before running any experiment, write down what I expect to happen. Most of my predictions were wrong, and figuring out why was the point.

## What's in here

| File | What it does |
|---|---|
| `Value.py` | The autograd engine. A `Value` holds a number, the operation that made it, and its gradient. Supports `+`, `-`, `*`, `tanh`, `relu`, and `backward_pass()` (topological sort, then the chain rule in reverse). Also holds `loss_function` (summed squared error). |
| `Nueron.py` | One neuron: a weight per input, a bias, and an activation. |
| `Layer.py` | A list of neurons that all receive the same inputs. |
| `Nueral_Network.py` | Stacks layers, runs the forward pass, and updates weights after backprop. The last layer is linear (no activation). Includes `count_dead_neurons()` for ReLU. |
| `dataset.py` | Generates the training data (see below). |
| `main.py` | Trains a network, evaluates it on held-out data, and saves a plot. Experiment settings are at the top. |
| `plot.py` | Plots the loss curve and the network's prediction against the true rule. |

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install matplotlib
python main.py
```

Change the settings at the top of `main.py` (target, layer sizes, epochs, seed) to run a different experiment. The activation is switched in `Nueron.forward_pass` (`.tanh()` or `.relu()`).

Training is slow, since every multiply and add is a Python object. A few hundred epochs on 400 examples takes a while.

## The data

500 points, each with 4 inputs drawn from [-1, 1]. 400 for training, 100 held out for testing. Two targets:

- **sine:** `sin(pi * x1) * x2`, smooth
- **triangle:** a triangle wave with the same peaks and period as the sine, but with sharp corners

`x3` and `x4` never affect the target. They're there as noise.

The **baseline** is the loss of always predicting the average target, ignoring the inputs. A network has only learned something if it beats that.

## What I found

All runs use the same seed, data, and network shape `[4, 4, 6, 8, 10, 12, 8, 4, 1]` unless noted.

1. **Zero initialization doesn't learn.** With every weight at 0, the loss sat at exactly 0.1618 every epoch. Each weight's gradient is multiplied by either a 0 activation or a 0 weight, so only the output bias ever moves.
2. **A 1-neuron layer is a bottleneck.** `[4, 2, 1, 2, 1]` learned in long plateaus and sudden drops, ending at a test loss of 0.0322. Wider networks learned almost immediately.
3. **A linear output layer helps.** Removing tanh from the last layer lowered test loss from 0.0064 to 0.0045 (100 epochs), and the network reached the peaks of the sine better.
4. **Tanh beat ReLU, and I tried to find out why** (400 epochs):

   | Test loss | sine | triangle |
   |---|---|---|
   | tanh | 0.0003 | 0.0006 |
   | ReLU | 0.0015 | 0.0010 |

   - *Smooth vs corners:* corners made tanh about 3x worse relative to the baseline, but didn't help ReLU. Not the full explanation.
   - *Dead ReLU neurons:* 7 of 52 were dead. A lower learning rate left only 3 dead, but the loss got 5x worse. Not the cause.
   - *Learning rate (0.003, 0.0085, 0.02):* dead neurons rose 3, 7, 10. None caught up with tanh. At 0.02, test loss was twice the training loss.
   - *Initialization:* not tested yet.

5. **Neither network extrapolates.** Outside the trained range, tanh flattens to a constant and ReLU continues in a straight line. Neither knows the wave repeats.

No overfitting showed up in any run with the full dataset: test loss stayed close to training loss, since the targets have no noise.

## Next

- Test He initialization for ReLU
- Move to PyTorch and Karpathy's makemore series