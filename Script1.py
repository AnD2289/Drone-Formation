import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the number of drones and their initial and final positions
num_drones = 5
initial_positions = np.random.rand(num_drones, 2) * 10  # Random initial positions
final_positions = np.array([[2, 2], [4, 4], [6, 2], [8, 4], [10, 2]])

# Create a figure and axes
fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)

# Initialize drone positions
drone_positions = initial_positions.copy()

# Plot initial drone positions as dots
drone_dots = ax.plot(initial_positions[:, 0], initial_positions[:, 1], 'ro')

# Update function for animation
def update(frame):
    global drone_positions
    if frame < 100:  # Number of frames for animation
        # Calculate the new drone positions based on a simple movement model (e.g., linear interpolation)
        alpha = frame / 100
        drone_positions = (1 - alpha) * initial_positions + alpha * final_positions
        drone_dots[0].set_data(drone_positions[:, 0], drone_positions[:, 1])
    else:
        drone_positions = final_positions
        drone_dots[0].set_data(drone_positions[:, 0], drone_positions[:, 1])
    return drone_dots

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(101), repeat=False)

# Display the animation
plt.show()
