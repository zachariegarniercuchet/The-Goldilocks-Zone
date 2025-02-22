import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def compute_total_parameters(layer_sizes):
    """
    Compute the total number of parameters D in a fully connected neural network.

    Args:
        layer_sizes: List of layer sizes, e.g., [input_size, hidden_size, ..., output_size]

    Returns:
        D: Total number of parameters
    """
    D = 0
    for i in range(len(layer_sizes) - 1):
        # Weights between layer i and layer i+1
        D += layer_sizes[i] * layer_sizes[i + 1]
        # Biases for layer i+1
        D += layer_sizes[i + 1]
    return D

# Define a simple function to initialize weights
def init_weights(shape, method='xavier'):
    if method == 'xavier':
        return nn.init.xavier_normal_(torch.empty(shape))
    elif method == 'he':
        return nn.init.kaiming_normal_(torch.empty(shape), nonlinearity='relu')

def generate_random_parameters(D, d, r):

    # Générer un point P sur la sphère de rayon r dans un espace de dimension D
    P = torch.randn(D)
    P = P / torch.norm(P) * r

    # Générer une matrice M de taille D x d avec des entrées gaussiennes
    M = torch.randn(D, d)

    # Orthogonaliser les colonnes de M en utilisant le processus de Gram-Schmidt
    M, _ = torch.linalg.qr(M)

    return P, M

def plot_2d_circle_and_line(ax, radius, P, M, tangency):
    # Tracer le cercle
    circle = plt.Circle((0, 0), radius, color='tab:blue', fill=False)
    ax.add_artist(circle)

    # Tracer la droite
    a = 10  # Étendre la ligne dans les deux directions
    line_x = [P[0].item() - a * M[0, 0].item(), P[0].item() + a * M[0, 0].item()]
    line_y = [P[1].item() - a * M[1, 0].item(), P[1].item() + a * M[1, 0].item()]
    ax.plot(line_x, line_y, color='tab:orange', label = np.round(tangency,4))

    # Tracer le point P
    ax.plot(P[0].item(), P[1].item(), 'ko')

    # Limites des axes
    ax.legend(fontsize = 16)
    ax.set_xlim(-radius-1, radius+1)
    ax.set_ylim(-radius-1, radius+1)
    ax.set_aspect('equal', 'box')
    ax.grid(True)

def plot_3d_sphere_and_line(ax, radius, P, M):

    # Tracer la droite
    a = 10  # Étendre la ligne dans les deux directions
    line_x = [P[0].item() - a * M[0, 0].item(), P[0].item() + a * M[0, 0].item()]
    line_y = [P[1].item() - a * M[1, 0].item(), P[1].item() + a * M[1, 0].item()]
    line_z = [P[2].item() - a * M[2, 0].item(), P[2].item() + a * M[2, 0].item()]
    ax.plot(line_x, line_y, line_z, color='tab:orange')


    # Tracer la sphère
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='tab:blue', alpha=0.3, edgecolor='k', shade=True)

    # Tracer le point P
    ax.scatter(P[0].item(), P[1].item(), P[2].item(), color='r')

    # Limites des axes
    ax.set_xlim(-radius-1, radius+1)
    ax.set_ylim(-radius-1, radius+1)
    ax.set_zlim(-radius-1, radius+1)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

def plot_3d_sphere_and_plane(ax, radius, P, M, tangency):

    # Calculer le vecteur normal au plan
    normal_vector = torch.cross(M[:, 0], M[:, 1])

    # Équation du plan : normal_vector · (x - P) = 0
    # Résoudre pour z : z = (d - ax - by) / c
    a, b, c = normal_vector[0].item(), normal_vector[1].item(), normal_vector[2].item()
    d = torch.dot(normal_vector, P)

    # Tracer le plan
    xx, yy = np.meshgrid(np.linspace(-radius, radius, 10), np.linspace(-radius, radius, 10))
    zz = (d - a * xx - b * yy) / c
    ax.plot_surface(xx, yy, zz, color='tab:orange', alpha=0.5, edgecolor='k', shade=True, label = np.round(tangency,4))



    # Tracer la sphère
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='tab:blue', alpha=0.3, edgecolor='k', shade=True)


    # Tracer le point P
    ax.scatter(P[0].item(), P[1].item(), P[2].item(), color='r')

    # Limites des axes
    ax.legend(fontsize = 16)
    ax.set_xlim(-radius-1, radius+1)
    ax.set_ylim(-radius-1, radius+1)
    ax.set_zlim(-radius-1, radius+1)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)