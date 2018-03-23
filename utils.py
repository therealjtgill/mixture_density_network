import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gaussian_mixture_contour(means, covariances, weights, x_bounds, y_bounds):

  num_gaussians = len(means)

  x_ = np.linspace(x_bounds[0], x_bounds[1], 100)
  y_ = np.linspace(y_bounds[0], y_bounds[1], 100)
  x_grid, y_grid = np.mgrid[x_bounds[0]:x_bounds[1]:0.01,
                            y_bounds[0]:y_bounds[1]:0.01]
  #x_grid, y_grid = np.meshgrid(x_, y_)
  grid = np.dstack([x_grid, y_grid])
  #grid = np.column_stack([x_grid, y_grid])
  #print("grid shape:", grid.shape)

  z = np.zeros((grid.shape[0:-1]))
  for i in range(num_gaussians):
    N = multivariate_normal(means[i], covariances[i])
    z += weights[i]*N.pdf(grid)

  # plt.figure()
  # plt.contourf(x_grid, y_grid, z)
  # plt.show()
  return z


def make_grid(x_bounds, y_bounds):

  x_ = np.linspace(x_bounds[0], x_bounds[1], 100)
  y_ = np.linspace(y_bounds[0], y_bounds[1], 100)
  x_grid, y_grid = np.mgrid[x_bounds[0]:x_bounds[1]:0.05,
                            y_bounds[0]:y_bounds[1]:0.05]

  return x_grid, y_grid


if __name__ == "__main__":

  means = [[-1, -2], [-.8, -2.5]]
  covariances = [[[.6,0], [0,.6]], [[.75, .1],[.1, .75]]]
  weights = [0.45, 0.55]

  x = [-2, 5]
  y = [-3, 6]

  x_ = np.linspace(x[0], x[1], 100)
  y_ = np.linspace(y[0], y[1], 100)
  x_grid, y_grid = np.mgrid[x[0]:x[1]:0.05,
                            y[0]:y[1]:0.05]

  plt.figure()

  z = gaussian_mixture_contour(means, covariances, weights, x, y)

  plt.contourf(x_grid, y_grid, z)

  means = [[3, 4], [3.4, 3.6]]
  covariances = [[[.6,.2], [.2,.6]], [[.7, .5],[.5, .7]]]
  weights = [0.65, 0.35]

  z += gaussian_mixture_contour(means, covariances, weights, x, y)

  plt.contourf(x_grid, y_grid, z)
  plt.show()