import random

from Nueral_Network import Nueral_Network
from Value import Value, loss_function
from dataset import make_dataset, TARGETS
from plot import plot_results

# ---- Experiment settings: change ONE at a time ----
TARGET = "triangle"      # "sine" (smooth) or "triangle" (corners)
ACTIVATION = "relu"      # just a label for the printout and plot; switch the
                         # activation itself in Nueron.forward_pass as before
SIZE = [4, 4, 6, 8, 10, 12, 8, 4, 1]
EPOCHS = 400
SEED = 0

train, test = make_dataset(TARGET)
random.seed(SEED)
nn = Nueral_Network(SIZE)

# Loss of the best constant output (the mean target): the bar to beat.
ys = [y for _, y in train]
mean_y = sum(ys) / len(ys)
baseline = sum((y - mean_y) ** 2 for y in ys) / len(ys)
print(f"target {TARGET}, activation {ACTIVATION}, size {SIZE}, baseline {baseline:.4f}")

train_losses = []
for epoch in range(EPOCHS):
    total_loss = 0.0
    for x, y in train:
        inputs = [Value(v) for v in x]
        target = [Value(y)]
        predicted = nn.forward(inputs)
        loss = loss_function(predicted, target)
        nn.back_pass(loss)
        total_loss += loss.value
    train_losses.append(total_loss / len(train))
    print(f"epoch {epoch}: avg train loss {train_losses[-1]:.4f}")

test_loss = 0.0
for x, y in test:
    predicted = nn.forward([Value(v) for v in x])
    test_loss += loss_function(predicted, [Value(y)]).value
test_loss /= len(test)

print(nn.count_dead_neurons(train))

# One summary line per run: copy these into your notes to compare.
print(f"RESULT target={TARGET} activation={ACTIVATION} "
      f"train={train_losses[-1]:.4f} test={test_loss:.4f} "
      f"typical_miss={test_loss ** 0.5:.3f}")

predict = lambda x: nn.forward([Value(v) for v in x])[0].value
plot_results(train_losses, predict, baseline=baseline,
             truth=TARGETS[TARGET],
             title=f"{TARGET} target, {ACTIVATION} activation",
             path=f"results_{TARGET}_{ACTIVATION}.png")
