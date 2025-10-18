import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
lambda_val = 1.0
nu = 0.01
v1, v2 = 1.0, 0.5
Tc = 1.0
k = 50.0
sc = (0.5, 0.5)

# Domaine
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
dx, dy = Lx/Nx, Ly/Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Temps
T = 0.1
Nt = int(input("Donner une valeur pour Nt : "))
dt = T / Nt

# Initialisation
U = np.zeros((Ny, Nx))
U0 = np.zeros_like(U)

# Fonction source
def f(t, x, y):
    d2 = (x - sc[0])**2 + (y - sc[1])**2
    return Tc * np.exp(-k * d2)

# Conditions de Dirichlet sur bords entrants
def apply_dirichlet(U):
    if v1 > 0: U[:, 0] = 0
    if v1 < 0: U[:, -1] = 0
    if v2 > 0: U[0, :] = 0
    if v2 < 0: U[-1, :] = 0
    return U

# Schéma d’Euler explicite
for n in range(Nt):
    Ux = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2*dx)
    Uy = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2*dy)
    Uxx = (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / dx**2
    Uyy = (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / dy**2
    F = f(n*dt, X, Y)
    U += dt * (-v1*Ux - v2*Uy + nu*(Uxx + Uyy) - lambda_val*U + F)
    U = apply_dirichlet(U)

# Erreur L2 (par rapport à U0 ici)
L2_U = np.sqrt(np.sum((U - U0)**2) * dx * dy)

# Norme du gradient
grad_U = np.sqrt(Ux**2 + Uy**2)

# Tracés
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
im0 = axs[0].imshow(U, extent=[0, Lx, 0, Ly], origin='lower')
axs[0].set_title("Solution numérique u")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow((U - U0)**2, extent=[0, Lx, 0, Ly], origin='lower')
axs[1].set_title(f"Erreur L² = {L2_U:.3e}")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(grad_U, extent=[0, Lx, 0, Ly], origin='lower')
axs[2].set_title("Norme du gradient |∇u|")
plt.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("figures_convection_diffusion.png", dpi=300)
plt.show()