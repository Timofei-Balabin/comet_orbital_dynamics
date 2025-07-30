import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


k = 1.0          
m0 = 1.0e6       
m1 = 10.0        
a1 = 1.0         
a2 = 2.0 * a1      
e2 = 0.8  
r2_0 = a2 * (1 - e2)  
v2_0 = np.sqrt(k * m0 * (1 + e2) / r2_0)  

def equations(t, y):
    x2, y2, vx2, vy2 = y
    r2 = np.sqrt(x2**2 + y2**2)
    
    omega1 = np.sqrt(k * m0 / a1**3)
    x1 = a1 * np.cos(omega1 * t)
    y1 = a1 * np.sin(omega1 * t)
    
    dx = x2 - x1
    dy = y2 - y1
    r12 = np.sqrt(dx**2 + dy**2)
    
    ax = -k * m0 * x2 / r2**3 - k * m1 * dx / r12**3
    ay = -k * m0 * y2 / r2**3 - k * m1 * dy / r12**3
    
    return [vx2, vy2, ax, ay]

t_span = (0, 10 * 2 * np.pi * np.sqrt(a2**3 / (k * m0)))  
sol = solve_ivp(equations, t_span, 
                 [r2_0, 0, 0, v2_0], 
                 method='RK45', 
                 rtol=1e-8,
                 dense_output=True)


def compute_energy(t):
    state = sol.sol(t)
    x2, y2, vx2, vy2 = state
    
    omega1 = np.sqrt(k * m0 / a1**3)
    x1 = a1 * np.cos(omega1 * t)
    y1 = a1 * np.sin(omega1 * t)
    
    r2 = np.sqrt(x2**2 + y2**2)  
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    
    v2 = np.sqrt(vx2**2 + vy2**2)
    kinetic = 0.5 * v2**2
    
    potential = -k * m0 / r2 - k * m1 / r12
    
    total_energy = kinetic + potential
    
    return kinetic, potential, total_energy


times = np.linspace(t_span[0], t_span[1], 500)
energies = np.array([compute_energy(t) for t in times])
kinetic_energies = energies[:, 0]
potential_energies = energies[:, 1]
total_energies = energies[:, 2]

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(sol.y[0], sol.y[1], label='Комета (вытянутая орбита)')
plt.plot(a1 * np.cos(np.linspace(0, 2 * np.pi, 100)), 
         a1 * np.sin(np.linspace(0, 2 * np.pi, 100)), 
         'r-', label='Внутренняя планета')
plt.scatter([0], [0], c='gold', s=300, label='Звезда')
plt.axis('equal')
plt.legend()
plt.grid()
plt.title('Движение кометы с вытянутой орбитой')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.plot(times, kinetic_energies, 'b-', label='Кинетическая энергия')
plt.plot(times, potential_energies, 'r-', label='Потенциальная энергия')
plt.plot(times, total_energies, 'g-', label='Полная энергия')
plt.xlabel('Время')
plt.ylabel('Энергия')
plt.title('Изменение энергии системы')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

initial_energy = total_energies[0]
final_energy = total_energies[-1]
relative_change = abs((final_energy - initial_energy) / initial_energy)

print(f"Относительное изменение полной энергии: {relative_change:.2e}")
