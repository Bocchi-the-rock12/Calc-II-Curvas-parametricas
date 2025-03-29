import numpy as np
import sympy as sp
import plotly.graph_objects as go
import mpmath as mp
import warnings

# Suprime warnings específicos (como o RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

mp.mp.dps = 50  # Increase numerical precision to 50 digits

# Define symbolic variable
t = sp.Symbol("t")

# 1. Graphical representation of the trajectory
def represent_trajectory(a, b, x, y, z, P, V):
    val_t = np.linspace(a, b, 100)
    fx = sp.lambdify(t, x, "numpy") if isinstance(x, sp.Basic) and x.has(t) else (lambda t_val: np.full_like(t_val, float(x)))
    fy = sp.lambdify(t, y, "numpy") if isinstance(y, sp.Basic) and y.has(t) else (lambda t_val: np.full_like(t_val, float(y)))
    fz = sp.lambdify(t, z, "numpy") if isinstance(z, sp.Basic) and z.has(t) else (lambda t_val: np.full_like(t_val, float(z)))

    val_x, val_y, val_z = fx(val_t), fy(val_t), fz(val_t)

    # Calcular o valor máximo entre os eixos x, y e z
    max_value = max(np.max(np.abs(val_x)), np.max(np.abs(val_y)), np.max(np.abs(val_z)))

    # Definir os limites dos eixos com base no valor máximo
    axis_limit = max_value * 1.1  # Adiciona uma margem de 10%

    fig = go.Figure()

    # Add the curve
    fig.add_trace(go.Scatter3d(x=val_x, y=val_y, z=val_z,
                               mode='lines', line=dict(color='black', width=4),
                               name='Parametric Curve'))

    # Add point P
    fig.add_trace(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]],
                               mode='markers', marker=dict(color='red', size=5),
                               name='Point P'))

    # Add velocity vector
    end_point = [(P[i] + V[i]) for i in range(3)]
    scale_factor = 0.2 * np.linalg.norm(V)  # Você pode ajustar esse fator conforme necessário
    fig.add_trace(go.Scatter3d(x=[P[0], end_point[0]], y=[P[1], end_point[1]], z=[P[2], end_point[2]],
                               mode='lines', line=dict(color='green', width=6),
                               name='Velocity Vector'))

    # Ajuste a escala do cone de acordo com a escala do gráfico
    fig.add_trace(go.Cone(x=[end_point[0]], y=[end_point[1]], z=[end_point[2]],
                          u=[V[0] * scale_factor / np.linalg.norm(V)],
                          v=[V[1] * scale_factor / np.linalg.norm(V)],
                          w=[V[2] * scale_factor / np.linalg.norm(V)],
                          sizemode='absolute', sizeref=scale_factor,
                          anchor='tip', colorscale=[[0, 'green'], [1, 'green']], showscale=False))


    # Add the plane perpendicular to the velocity vector at P
    V_np = np.array(V, dtype=float)
    if np.allclose(V_np/np.linalg.norm(V_np), [0, 0, 1]):
        arbitrary = np.array([0, 1, 0])
    else:
        arbitrary = np.array([0, 0, 1])
    u = np.cross(V_np, arbitrary)
    u = u / np.linalg.norm(u)
    w = np.cross(V_np, u)
    w = w / np.linalg.norm(w)
    # Ajustar o tamanho do plano com base no valor máximo dos eixos
    s_vals = np.linspace(-axis_limit, axis_limit, 10)  # Ajuste os valores de s
    t_vals = np.linspace(-axis_limit, axis_limit, 10)  # Ajuste os valores de t
    S, T = np.meshgrid(s_vals, t_vals)
    S, T = np.meshgrid(s_vals, t_vals)
    X_plane = P[0] + S*u[0] + T*w[0]
    Y_plane = P[1] + S*u[1] + T*w[1]
    Z_plane = P[2] + S*u[2] + T*w[2]

    fig.add_trace(go.Surface(x=X_plane, y=Y_plane, z=Z_plane, opacity=0.5,
                             colorscale=[[0, 'blue'], [1, 'blue']],
                             name='Perpendicular Plane',
                             showscale=False))

    #fig.update_layout(
        #scene=dict(
         #   xaxis_title='x',
          #  yaxis_title='y',
           # zaxis_title='z',
            #aspectmode='auto'
        #),
        #title='Parametric Trajectory',
        #margin=dict(l=0, r=0, b=0, t=30)
    #)

    # Ajustar os limites dos eixos com base no valor máximo
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            xaxis=dict(range=[-axis_limit, axis_limit]),
            yaxis=dict(range=[-axis_limit, axis_limit]),
            zaxis=dict(range=[-axis_limit, axis_limit]),
            aspectmode='auto'
        ),
        title='Parametric Trajectory',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

# 2. General expression of the velocity vector
def velocity_vector_expression(x, y, z):
    return [sp.diff(x, t) if isinstance(x, sp.Basic) and x.has(t) else 0,
            sp.diff(y, t) if isinstance(y, sp.Basic) and y.has(t) else 0,
            sp.diff(z, t) if isinstance(z, sp.Basic) and z.has(t) else 0]

# 3. Length of the curve
def curve_length(a, b, r):
    mod_r = sp.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    L_total = sp.integrate(mod_r, (t, a, b)).evalf()
    return mod_r, L_total

# 4. Calculation of point and velocity vector
def calculate_point_and_velocity(x, y, z, r, t0):
    fx = sp.lambdify(t, x, "numpy") if isinstance(x, sp.Basic) and x.has(t) else (lambda t_val: float(x))
    fy = sp.lambdify(t, y, "numpy") if isinstance(y, sp.Basic) and y.has(t) else (lambda t_val: float(y))
    fz = sp.lambdify(t, z, "numpy") if isinstance(z, sp.Basic) and z.has(t) else (lambda t_val: float(z))

    dx = sp.lambdify(t, r[0], "numpy") if isinstance(r[0], sp.Basic) and r[0].has(t) else (lambda t_val: float(r[0]))
    dy = sp.lambdify(t, r[1], "numpy") if isinstance(r[1], sp.Basic) and r[1].has(t) else (lambda t_val: float(r[1]))
    dz = sp.lambdify(t, r[2], "numpy") if isinstance(r[2], sp.Basic) and r[2].has(t) else (lambda t_val: float(r[2]))

    P = [fx(t0), fy(t0), fz(t0)]
    V = [dx(t0), dy(t0), dz(t0)]
    return P, V

def main():
    # Get user input for x, y, z, t0, a, and b
    x_input = input("Enter the expression for x (e.g., cos(t)): ")
    y_input = input("Enter the expression for y (e.g., sin(t)): ")
    z_input = input("Enter the expression for z (e.g., 3*cos(t)): ")
    a = float(input("Enter the value of a: "))
    b = float(input("Enter the value of b: "))
    t0 = float(input("Enter the value of t0: "))

    # Check if t0 is within the range [a, b]
    if t0 < a or t0 > b:
        print("Invalid t0. It must be between a and b.")
        return

    # Convert input strings to sympy expressions
    x = sp.sympify(x_input)
    y = sp.sympify(y_input)
    z = sp.sympify(z_input)

    try:
      val_t = np.linspace(a, b, 100)
      # Tenta avaliar as funções em todos os pontos do intervalo
      fx = sp.lambdify(t, x, "numpy")
      fy = sp.lambdify(t, y, "numpy")
      fz = sp.lambdify(t, z, "numpy")

      # Avalia os valores das funções nos pontos de t
      val_x = fx(val_t)
      val_y = fy(val_t)
      val_z = fz(val_t)

      # Verifica se algum resultado é inválido (ex: NaN, infinito)
      if np.any(np.isnan(val_x)) or np.any(np.isnan(val_y)) or np.any(np.isnan(val_z)):
        raise ValueError("A função gera valores indefinidos no intervalo.")
      if np.any(np.isinf(val_x)) or np.any(np.isinf(val_y)) or np.any(np.isinf(val_z)):
        raise ValueError("A função gera valores infinitos no intervalo.")

    except Exception as e:
      print(f"Erro ao avaliar a função no domínio: {e}")
      return

    # Calculate the velocity vector, curve length, and point at t0
    r = velocity_vector_expression(x, y, z)
    mod_r, L_total = curve_length(a, b, r)
    L_arc = sp.integrate(mod_r, (t, a, t0)).evalf()
    P, V = calculate_point_and_velocity(x, y, z, r, t0)

    # Represent the trajectory
    represent_trajectory(a, b, x, y, z, P, V)

    P = [float(coord) for coord in P]

    # Print the results
    print(f'Coordinates of point P at time t0 = {t0}: ({P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f})')
    print(f'Total length of the curve: {L_total:.4f}')
    print(f'Arc length up to t0: {L_arc:.4f}')

main()
