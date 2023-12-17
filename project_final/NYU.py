import numpy as np
import matplotlib.pyplot as plt
# Adjusting the code to make all letters 'N', 'Y', and 'U' equally thick

# Create an empty 200x200 matrix with ones
NYU = np.ones((200, 200))

# Define the coordinates for a thicker 'N'
n_coords = [
    (j, i) for i in range(50, 130) for j in range(30, 35)  # Left vertical line
] + [
    (j, -(40 + (2 * i - 30))) for i in range(30, 70) for j in range(i-1, i + 5)  # Diagonal
] + [
    (j, i) for i in range(50, 130) for j in range(70, 75)  # Right vertical line
]

# Define the coordinates for a thicker 'Y'
y_coords = [
    (j, i) for i in range(40, 103) for j in range(110, 115)  # Vertical line
] + [
    (j, 100 + (i - 110)) for i in range(110, 140) for j in range(i, i + 5)  # Right diagonal
] + [
    (j, 130 - (i - 80)) for i in range(80, 110) for j in range(i, i + 5)  # Left diagonal
]

# Define the coordinates for a thicker 'U'
u_coords = [
    (j, i) for i in range(55, 130) for j in range(150, 155)  # Left vertical line
] + [
    (i, j) for i in range(150, 190) for j in range(55, 60)  # Bottom horizontal line
] + [
    (j, i) for i in range(55, 130) for j in range(190, 195)  # Right vertical line
]

# Set the specified coordinates to -1
for coord in n_coords + y_coords + u_coords:
    NYU[coord] = -1

# Display the matrix
plt.imshow(NYU, cmap='gray')
plt.axis('off')
plt.show()
