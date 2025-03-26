import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Definir variável simbólica
t = sp.Symbol("t")

# Valores a alterar
a = -np.pi
b = 2*np.pi
x = t
y = sp.sin(t)
z = 2*sp.cos(t)
t0 = np.pi  # Escolha para ser distinto do ponto médio

############# 1. Representação gráfica da trajetória ###########
def representa_trajetoria(a, b, x, y, z, P, V):
    val_t = np.linspace(a, b, 100)
    fx = sp.lambdify(t, x, "numpy")
    fy = sp.lambdify(t, y, "numpy")
    fz = sp.lambdify(t, z, "numpy")

    val_x = fx(val_t)
    val_y = fy(val_t)
    val_z = fz(val_t)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(val_x, val_y, val_z, '-k', label='Curva Paramétrica')
    ax.scatter(*P, color='r', label='Ponto P', s=50)
    ax.quiver(P[0], P[1], P[2], V[0], V[1], V[2], color='g', length=1.0, linewidth=2, arrow_length_ratio=0.2, label='Vetor Velocidade')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.show()

########### 2. Expressão geral do vetor velocidade ##########
def expressao_vetor_velocidade(x, y, z):
    dx = sp.diff(x, t)
    dy = sp.diff(y, t)
    dz = sp.diff(z, t)
    return [dx, dy, dz]

################# 3. Comprimento da curva #################
def comprimento_curva(a, b, r):
    mod_r = sp.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    L = sp.integrate(mod_r, (t, a, b))
    L = sp.N(L)
    return mod_r, L

###### 4. Cálculo do ponto e vetor velocidade #########
def calcular_ponto_e_vetor(x, y, z, r, t0):
    fx = sp.lambdify(t, x)
    fy = sp.lambdify(t, y)
    fz = sp.lambdify(t, z)
    dx = sp.lambdify(t, r[0], "numpy")
    dy = sp.lambdify(t, r[1], "numpy")
    dz = sp.lambdify(t, r[2], "numpy")
    P = [fx(t0), fy(t0), fz(t0)]
    V = [dx(t0), dy(t0), dz(t0)]
    return P, V

def main():
    r = expressao_vetor_velocidade(x, y, z)
    mod_r, L = comprimento_curva(a, b, r)
    arc_length_t0 = sp.integrate(mod_r, (t, a, t0)).evalf()
    P, V = calcular_ponto_e_vetor(x, y, z, r, t0)
    representa_trajetoria(a, b, x, y, z, P, V)

    # Mostrar as coordenadas do ponto P
    print(f'Coordenadas do ponto P no tempo t0 = {t0}: ({P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f})')

    print(f'Comprimento total da curva: {L:.4f}')
    print(f'Comprimento do arco até t0: {arc_length_t0:.4f}')

main()
