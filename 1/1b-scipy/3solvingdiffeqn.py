import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation dy/dt = -2y
def dydt(t, y):
    return -2 * y

# Initial condition
y0 = [1]

# Time points where solution is computed
t = np.linspace(0, 5, 100)

# Solve the ODE
solution = solve_ivp(dydt, [0, 5], y0, t_eval=t)

# Plot the solution
plt.plot(solution.t, solution.y[0])
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -2y')
plt.show()
