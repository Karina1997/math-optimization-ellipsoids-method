import matplotlib.pyplot as plt

def plot_results(losses_ell, losses_sgd, batch_sz, lim):
  if len(losses_ell) != 0:
    plt.plot(losses_ell, label="Ellipsoid method", color="C0")

  if len(losses_sgd) != 0:
    plt.plot(losses_sgd, label= "SGD", color="C1")

  plt.grid()
  plt.legend()
  plt.ylim(top=lim)
  plt.ylabel("Loss")
  plt.xlabel("Iteration")
  plt.title("Batch size = " + str(batch_sz))
  plt.show()