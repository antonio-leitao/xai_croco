import matplotlib.pyplot as plt
import numpy as np

def calculate_point(weights, pad=0.1):
    # Calculate the normalized coordinates of the point with padding
    total_weight = sum(weights.values()) + 3*pad
    x = (weights['proximity']+pad + (weights['validity']+pad) / 2) / total_weight
    y = (weights['validity']+pad) * np.sqrt(3) / (2 * total_weight)
    tot =  sum(weights.values())
    coors = [np.round(w/tot,1) for w in weights.values()]
    return x,y,coors
    
def draw_triangle(vertices,ax):
    # Color the inside of the triangle
    triangle = np.array([vertices['validity'], vertices['proximity'], vertices['robustness'], vertices['validity']])
    ax.fill(triangle[:, 0], triangle[:, 1], color='powderblue',alpha=0.05)

    # Plot the triangle
    ax.plot([vertices['validity'][0], vertices['proximity'][0]],
             [vertices['validity'][1], vertices['proximity'][1]], 'k',linewidth=2.0)
    ax.plot([vertices['proximity'][0], vertices['robustness'][0]],
             [vertices['proximity'][1], vertices['robustness'][1]], 'k',linewidth=2.0)
    ax.plot([vertices['robustness'][0], vertices['validity'][0]],
             [vertices['robustness'][1], vertices['validity'][1]], 'k',linewidth=2.0)

    ax.text(vertices['validity'][0], vertices['validity'][1]-0.1,
             f'Robustness', ha='center', va='bottom',fontsize=10)
    ax.text(vertices['proximity'][0], vertices['proximity'][1]-0.1,
             f'Proximity', ha='center', va='bottom',fontsize=10)
    ax.text(vertices['robustness'][0], vertices['robustness'][1]+0.02,
             f'Validity', ha='center', va='bottom',fontsize=10)

def tri_plot(weights,ax):
    vertices = {'validity': [0, 0], 'proximity': [1, 0], 'robustness': [0.5, np.sqrt(3) / 2]}
    x,y,coors = calculate_point(weights)
    
    draw_triangle(vertices, ax)

    # Plot the point inside the triangle
    ax.plot(x, y, ms=12,marker="o")  # Plot the point
    ax.axis("equal")
    ax.axis('off')
    ax.text(0.5,-0.05,coors,fontsize=6,ha="center", va="center")
    
def tri_plot_arrow(start,stop,ax):
    vertices = {'validity': [0, 0], 'proximity': [1, 0], 'robustness': [0.5, np.sqrt(3) / 2]}
    draw_triangle(vertices, ax)

    x1, y1,_ = calculate_point(start)
    x2, y2,_ = calculate_point(stop)

    # Plot the points
    ax.plot(x1, y1, ms=12,marker="o", color="#1f77b4")  # Red point
    ax.plot(x2, y2, ms=12,marker="o", color="#1f77b4")  # Blue point

    # Draw an arrow between the points
    ax.arrow(x1, y1, x2-x1, y2-y1, color='black', length_includes_head=True,
              head_width=0.04,head_length=0.08)
    ax.axis("equal")
    ax.axis('off')

def lerp(start_point, end_point, n_samples):
    dimensions = len(start_point)
    if dimensions != len(end_point):
        raise ValueError("Start and end points must have the same number of dimensions")

    # Generate interpolation weights
    weights = [i / (n_samples - 1) for i in range(n_samples)]
    
    # Perform linear interpolation
    interpolated_points = []
    for weight in weights:
        interpolated_point = [(1 - weight) * start + weight * end for start, end in zip(start_point, end_point)]
        # Normalize the point to ensure the sum of elements is equal to 1
        sum_elements = sum(interpolated_point)
        interpolated_point = [element / sum_elements for element in interpolated_point]
        interpolated_points.append(interpolated_point)

    return interpolated_points