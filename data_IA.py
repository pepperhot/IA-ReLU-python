import numpy as np
import matplotlib.pyplot as plt

def vortex(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.15
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


def square(points, classes):
    grid_size = int(np.sqrt(classes))
    X = np.zeros((points * classes, 2))
    y = np.repeat(np.arange(classes), points)
    for class_number in range(classes):
        row, col = divmod(class_number, grid_size)
        x_min, x_max = col / grid_size, (col + 1) / grid_size
        y_min, y_max = row / grid_size, (row + 1) / grid_size
        ix = slice(points * class_number, points * (class_number + 1))
        X[ix, 0] = np.random.rand(points) * (x_max - x_min) + x_min
        X[ix, 1] = np.random.rand(points) * (y_max - y_min) + y_min
    return X, y

def hearth(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        t = np.linspace(0, 2 * np.pi, points)
        size = 1 - class_number * 0.03
        X[ix, 0], X[ix, 1] = size * (16 * np.sin(t)**3),size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        y[ix] = class_number
    return X, y

def triangles(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        size = 1 - class_number * 0.05
        vertices = np.array([
            [0, size],
            [-size * np.sqrt(3)/2, -size/2],
            [size * np.sqrt(3)/2, -size/2]])
        t = np.random.rand(points)
        sides = np.random.randint(0, 3, points)
        X[ix] = (1 - t[:, None]) * vertices[sides] + t[:, None] * vertices[(sides + 1) % 3]
        y[ix] = class_number
    return X, y

def star(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    
    for class_number in range(classes):
        ix = slice(points * class_number, points * (class_number + 1))
        t = np.linspace(0, 2 * np.pi, points, endpoint=False)
        size = class_number * 0.03
        r = size * (1 + 0.5 * np.sin(5 * t))  # Forme d'étoile
        X[ix, 0] = r * np.cos(t)
        X[ix, 1] = r * np.sin(t)
        y[ix] = class_number
    
    return X, y
def fractal(points,classes):
    x_vals = np.linspace(-2, 1, points)
    y_vals = np.linspace(-1.5, 1.5, points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros(X.shape)
    
    for i in range(points):
        for j in range(points):
            c = complex(X[i, j], Y[i, j])
            z = 0
            for n in range(classes):
                if abs(z) > 2:
                    Z[i, j] = n
                    break
                z = z*z + c
            else:
                Z[i, j] = classes
    
    return X, Y, Z

"""print("Attention ça arrive !")
#X, y = vortex(300, 4)
#X, y = square(300, 182)
#X, y = hearth(300, 33)
#X, y = triangles(300, 10)
#X, Y, Z = fractal(800, 100)
#X, y = star(300, 10)


plt.scatter(X[: , 0], X[: , 1])
plt.show()


plt.scatter(X[:, 0], X[:, 1], c=y, cmap="inferno", vmin=-2, vmax=np.max(y))
plt.show()
"""
'''
# Aplatir les matrices X et Y pour qu'elles deviennent des vecteurs
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Affichage des points générés (pour afficher l'ensemble Mandelbrot)
plt.scatter(X_flat, Y_flat, c=Z_flat, cmap="inferno", s=0.5)  # Petit point pour plus de précision
plt.show()
'''
