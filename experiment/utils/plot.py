import matplotlib.pyplot as plt

def plot_results(losses_sgd_small_batch_size, losses_ell, losses_sgd, batch_sz, batch_sz_sgd, lim = -1):
  if len(losses_ell) != 0:
    plt.plot(losses_ell, label="Ellipsoid method, batch size = " + str(batch_sz), color="C0")

  if len(losses_sgd) != 0:
    plt.plot(losses_sgd, label= "SGD, batch size = " + str(batch_sz), color="C1")

  if len(losses_sgd_small_batch_size) != 0:
    plt.plot(losses_sgd_small_batch_size, linestyle = '--', label= "SGD, batch size = " + str(batch_sz_sgd), color="C2")

  plt.grid()
  plt.legend()
  if lim != -1:
    plt.ylim(bottom=0.58, top=lim)
  plt.ylabel("Loss")
  plt.xlabel("Iteration")
  plt.title("Test loss on iterations")
  plt.show()
