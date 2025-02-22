import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import generate_random_parameters, plot_2d_circle_and_line, plot_3d_sphere_and_line, plot_3d_sphere_and_plane, init_weights


def plot_distribution_norms_init(shape = (1000, 1000)):
    # Generate weights
    xavier_weights = init_weights(shape, method='xavier')
    he_weights = init_weights(shape, method='he')

    # Compute norms
    xavier_norms = torch.norm(xavier_weights, dim=1)
    he_norms = torch.norm(he_weights, dim=1)

    # Plot the distribution of norms
    plt.figure(figsize=(8,5))
    sns.histplot(xavier_norms.numpy(), label='Xavier', kde=True, color='blue', bins=50)
    sns.histplot(he_norms.numpy(), label='He', kde=True, color='red', bins=50)
    plt.legend()
    plt.xlabel('Norm of Weight Vectors')
    plt.ylabel('Density')
    plt.title('Weight Norms under Xavier vs He Initialization')
    plt.show()

def compute_tangents(D, d, r, num_trials, plot=False, P=None, M = None):
    tangents = []

    # Configuration et traçage des figures si plot est True
    if plot:
        if D == 2 and d == 1:
            c = 3
            fig, axes = plt.subplots(c, c, figsize=(16, 16))
            for i in range(c**2):
                P, M = generate_random_parameters(D, d, r)
                tangency = compute_tangents(2,1,r,1,False, P,M)
                plot_2d_circle_and_line(axes[i // c, i % c], r, P, M, tangency[0])
        elif D == 3 and d == 1:
            c = 2
            fig, axes = plt.subplots(c, c, figsize=(12, 12), subplot_kw={'projection': '3d'})
            for i in range(c**c):
                P, M = generate_random_parameters(D, d, r)
                plot_3d_sphere_and_line(axes[i // c, i % c], r, P, M)
        elif D == 3 and d == 2:
            c = 2
            fig, axes = plt.subplots(c, c, figsize=(12, 12), subplot_kw={'projection': '3d'})
            for i in range(c**2):
                P, M = generate_random_parameters(D, d, r)
                tangency = compute_tangents(3,2,r,1,False, P,M)
                plot_3d_sphere_and_plane(axes[i // c, i % c], r, P, M, tangency[0])
        plt.tight_layout()
        plt.show()

    # Boucle principale pour calculer les tengances
    for _ in range(num_trials):

        if P == None and M == None :
            P, M = generate_random_parameters(D, d, r)

        # Calculer la similarité cosinus entre chaque colonne de M et P
        cosine_similarities = torch.mv(M.t(), P) / (torch.norm(M, dim=0) * torch.norm(P))

        # Calculer la norme L2 du vecteur de similarités cosinus
        tangent = torch.norm(cosine_similarities)
        tangents.append(tangent.item())

    return tangents


def visualization_2(r=1, 
                    num_trials = 50, 
                    D_values =  range(10, 300, 10),
                    d_values = [1, 10, 50, 100]):

    torch.manual_seed(41)

    # Dictionnaire pour stocker les moyennes des tangentes pour chaque valeur de d
    mean_tangents = {d: [] for d in d_values}

    # Boucle principale pour différentes valeurs de D
    for D in D_values:
        for d in d_values:
            # Ne pas calculer si d >= D
            if d >= D:
                continue
            tangent = compute_tangents(D, d, r, num_trials)
            mean_tangent = np.mean(tangent)
            mean_tangents[d].append(mean_tangent)

    # Tracer le graphe des tangents en fonction de D pour chaque valeur de d
    plt.figure(figsize=(12, 8))
    for d in d_values:
        # Ajouter des valeurs NaN pour les valeurs manquantes où d >= D
        missing_count = sum(1 for D in D_values if d >= D)
        mean_tangents[d] = [np.nan] * missing_count + mean_tangents[d]
        plt.plot(D_values, mean_tangents[d], marker='o', linestyle='-', label=f'd={d}')

    plt.title('tangency of an hyperplane d on an hypershere D')
    plt.xlabel('Dimension D')
    plt.ylabel('Tangency metrics')
    plt.legend()
    plt.grid(True)
    plt.show()



def visualization_1(D,d,r = 5, num_trials = 1, plot = True):

    torch.manual_seed(47)

    # Calculer les tangents
    tangent = compute_tangents(D, d, r, num_trials, plot)

    # Calculer les statistiques
    mean_tangent = np.mean(tangent)
    std_tangent = np.std(tangent)

    return #mean_tangent, std_tangent