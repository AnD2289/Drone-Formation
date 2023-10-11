import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the number of drones and their initial and final positions
num_drones = 5
initial_positions = np.random.rand(num_drones, 2) * 10  # Random initial positions
final_positions = np.array([[2, 2], [4, 4], [6, 2], [8, 4], [10, 2]])

# Create a figure and axes for drone tracking
fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)

# Initialize drone positions and set a learning rate
drone_positions = initial_positions.copy()
learning_rate = 0.1  # Adjust the learning rate as needed

# Plot initial drone positions as dots
drone_dots = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c='red', s=100)  # Increase the marker size for better visibility

# Initialize variables for gradient tracking
gradient_norms = []

# Function to calculate the gradient and update positions using gradient descent
def update_positions():
    global drone_positions
    gradients = -learning_rate * (drone_positions - final_positions)
    drone_positions += gradients
    return gradients

# Update function for drone tracking
def update_drone_tracking(frame):
    if frame < 100:  # Number of frames for drone tracking
        update_positions()
        drone_dots.set_offsets(drone_positions)  # Update drone positions
        # Update the animation plot with the iteration count
        ax.set_title(f"Iteration: {frame}")

        gradient_norm = np.linalg.norm(update_positions())
        gradient_norms.append(gradient_norm)

    return drone_dots,

# Create the animation for drone tracking
ani_drone_tracking = animation.FuncAnimation(fig, update_drone_tracking, frames=range(101), repeat=False)

# Display the drone tracking animation and gradient tracking plot
plt.show()

# Create a separate figure for gradient tracking
fig_gradient, ax_gradient = plt.subplots()
ax_gradient.set_xlabel("Iteration")
ax_gradient.set_ylabel("Gradient Norm")

# Plot gradient norm
ax_gradient.plot(range(len(gradient_norms)), gradient_norms, 'b-')
plt.show()
