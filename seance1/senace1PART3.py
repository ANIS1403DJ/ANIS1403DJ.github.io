import numpy as np
import matplotlib.pyplot as plt

# === PARAMÈTRES PHYSIQUES ===
L = 1.0
V = 1.0
nu = 0.01
lam = 1.0
ul = 1.0     # Dirichlet à gauche
g = -0.5     # Neumann à droite (du/dx = g)

# === PARAMÈTRES NUMÉRIQUES ===
NX = 100
dx = L / (NX - 1)
x = np.linspace(0, L, NX)
dt = dx**2 / (V * dx + 4 * nu + dx**2)  # CFL
NT = 5000
eps = 1e-4

# === CONDITION INITIALE COMPATIBLE ===
u = ul + g * x.copy()

# === TERME SOURCE f(x) ===
def f(x):
    return np.zeros_like(x)  # exemple sans source

# === BOUCLE TEMPORELLE ===
res0 = 1.0
for n in range(NT):
    u_new = u.copy()
    res = 0.0
    for j in range(1, NX - 1):
        ux = (u[j+1] - u[j-1]) / (2 * dx)
        uxx = (u[j+1] - 2*u[j] + u[j-1]) / dx**2
        rhs = -V * ux + nu * uxx - lam * u[j] + f(x[j])
        u_new[j] = u[j] + dt * rhs
        res += abs(dt * rhs)

    # CL gauche : Dirichlet
    u_new[0] = ul

    # CL droite : Neumann
    u_new[-1] = u_new[-2] + g * dx

    # Convergence
    if n == 0:
        res0 = res
    if res / res0 < eps:
        break

    u = u_new.copy()

# === AFFICHAGE ===
plt.plot(x, u, label="Solution numérique")
plt.plot(x, ul + g * x, '--', label="Condition initiale")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution stationnaire ADRS 1D")
plt.legend()
plt.grid(True)
plt.show()