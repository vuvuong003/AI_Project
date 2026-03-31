import numpy as np
import matplotlib.pyplot as plt

## Grid in matplotlib

rows, cols = 50, 50
grid = np.zeros((rows, cols))

# obstacles
grid[5:8, 10:15] = 1  

# Place fire starting points
grid[0, 0] = 2  # Fire starts at top-left

#Safe Zone
grid[19, 29] = 3

# Plot
plt.imshow(grid, cmap='hot', origin='upper')
plt.colorbar()
plt.show()