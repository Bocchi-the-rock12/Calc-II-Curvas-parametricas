import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variable
t = sp.Symbol("t")

# Default values for user modification
x = t
y = sp.sin(t)
z = 2 * sp.cos(t)

def get_user_input(prompt, default):
    while True:
        user_input = input(f"{prompt} (default: {default}): ")
        if user_input.strip() == "":
            return default
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Get user-defined values
a = get_user_input("Enter the start value of t (a)", -np.pi)
b = get_user_input("Enter the end value of t (b)", 2 * np.pi)
t0 = get_user_input("Enter the specific point t0", np.pi)

############# 1. Graphical representation of the trajectory ###########
def representa_trajetoria(a, b, x, y, z, P, V, plano=False):
    val_t = np.linspace(a, b, 100)
    fx = sp.lambdify(t, x, "numpy")
    fy = sp.lambdify(t, y, "numpy")
    fz = sp.lambdify(t, z, "numpy")

    val_x = fx(val_t)
    val_y = fy(val_t)
    val_z = fz(val_t)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the curve
    ax.plot(val_x, val_y, val_z, '-k', label='Parametric Curve')
    ax.scatter(*P, color='r', label='Point P', s=50)

    # Normalize velocity vector for visualization
    V_norm = np.array(V) / np.linalg.norm(V)
    ax.quiver(P[0], P[1], P[2], V_norm[0], V_norm[1], V_norm[2],
              color='g', length=1.0, linewidth=2, arrow_length_ratio=0.2, label='Velocity Vector')

    # Plane representation
    if plano and V[2] != 0:
        normal = np.array(V)
        d = -np.dot(normal, P)
        xx, yy = np.meshgrid(np.linspace(P[0] - 2, P[0] + 2, 10), np.linspace(P[1] - 2, P[1] + 2, 10))
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

        # Plot plane with transparency
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='b')

    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    # Adjust graph limits for better visualization
    ax.set_xlim([min(val_x) - 1, max(val_x) + 1])
    ax.set_ylim([min(val_y) - 1, max(val_y) + 1])
    ax.set_zlim([min(val_z) - 1, max(val_z) + 1])

    # Enable interactive rotation
    plt.ion()
    plt.show()
    input("Press Enter to close the plot...")
    plt.close()

########### 2. General expression of velocity vector ##########
def expressao_vetor_velocidade(x, y, z):
    dx = sp.diff(x, t)
    dy = sp.diff(y, t)
    dz = sp.diff(z, t)
    return [dx, dy, dz]

################# 3. Curve length #################
def comprimento_curva(a, b, r):
    mod_r = sp.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    L = sp.integrate(mod_r, (t, a, b))
    L = sp.N(L)  # Ensure numerical evaluation
    return mod_r, L

###### 4. Compute point and velocity vector #########
def calcular_ponto_e_vetor(x, y, z, r, t0):
    fx = sp.lambdify(t, x, "numpy")
    fy = sp.lambdify(t, y, "numpy")
    fz = sp.lambdify(t, z, "numpy")
    dx = sp.lambdify(t, r[0], "numpy")
    dy = sp.lambdify(t, r[1], "numpy")
    dz = sp.lambdify(t, r[2], "numpy")

    P = [fx(t0), fy(t0), fz(t0)]
    V = [dx(t0), dy(t0), dz(t0)]
    return P, V

def main():
    r = expressao_vetor_velocidade(x, y, z)
    mod_r, L = comprimento_curva(a, b, r)
    arc_length_t0 = sp.N(sp.integrate(mod_r, (t, a, t0)))  # Ensure numerical evaluation
    P, V = calcular_ponto_e_vetor(x, y, z, r, t0)

    representa_trajetoria(a, b, x, y, z, P, V, plano=True)

    # Display results
    print(f'Point P at time t0 = {t0}: ({P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f})')
    print(f'Total curve length: {L:.4f}')
    print(f'Arc length up to t0: {arc_length_t0:.4f}')

if __name__ == "__main__":
    main()


