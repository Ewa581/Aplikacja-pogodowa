import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

x_values = []
y_values = []


plt.figure(figsize=(10, 6))
plt.title('Losowe liczby (x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)


def update_graph(frame):
   
    x = random.uniform(0, 100)  
    y = random.uniform(0, 100)  
    
    x_values.append(x)
    y_values.append(y)
    
    plt.cla()
    plt.scatter(x_values, y_values, color='blue')
    plt.title(f'Losowe liczby (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    if len(x_values) >= 50:
        ani.event_source.stop()

ani = FuncAnimation(plt.gcf(), update_graph, interval=3000)  

plt.tight_layout()
plt.show()