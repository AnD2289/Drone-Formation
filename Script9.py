import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the number of drones
num_drones = 20

# Create a figure and axes for drone tracking
fig, ax = plt.subplots()
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)

# Set the positions for the 'VT' symbol
vt_positions = np.array([
    [2, 5], [3, 5], [4, 5],
    [2, 4], [3, 4], [4, 4],
    [2, 3], [4, 3],
    [2, 2], [4, 2],
])

# Initialize drone positions randomly
drone_positions = np.random.rand(num_drones, 2) * 6

# Set a learning rate
learning_rate = 0.1

# Plot initial drone positions as dots
drone_dots = ax.scatter(drone_positions[:, 0], drone_positions[:, 1], c='red', s=100)

# Initialize variables for gradient tracking
gradient_norms = []

# Function to calculate the gradient and update positions using gradient descent
def update_positions():
    global drone_positions
    gradients = -learning_rate * (drone_positions - vt_positions)
    drone_positions += gradients
    return gradients

# Convergence criterion
def has_converged():
    return np.all(np.linalg.norm(drone_positions - vt_positions, axis=1) < 0.01)

# Update function for drone tracking
def update_drone_tracking(frame):
    if frame < 100 and not has_converged():
        update_positions()
        drone_dots.set_offsets(drone_positions)
        ax.set_title(f"Iteration: {frame}")
        gradient_norm = np.linalg.norm(update_positions())
        gradient_norms.append(gradient_norm)

    return drone_dots,

# Create the animation for drone tracking
ani_drone_tracking = animation.FuncAnimation(fig, update_drone_tracking, frames=range(101), repeat=False)

# Display the drone tracking animation
plt.show()

# Create a separate figure for gradient tracking
fig_gradient, ax_gradient = plt.subplots()
ax_gradient.set_xlabel("Iteration")
ax_gradient.set_ylabel("Gradient Norm")

# Plot gradient norm
ax_gradient.plot(range(len(gradient_norms)), gradient_norms, 'b-')
plt.show()
