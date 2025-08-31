import random

# Objective function
def objective_function(x):
    # objective function: maximum at x = 5
    return -x**2 + 10*x

# Hill Climbing Algorithm
def hill_climb(start, step_size, max_iterations):
    current_x = start
    current_value = objective_function(current_x)

    for i in range(max_iterations):
        # Generate one random neighbor
        next_x = current_x + random.uniform(-step_size, step_size)
        next_value = objective_function(next_x)

        # If next solution is better, move to it
        if next_value > current_value:
            current_x = next_x
            current_value = next_value
            print(f"Iteration {i+1}: x = {current_x:.4f}, f(x) = {current_value:.4f}")
        else:
            print(f"Iteration {i+1}: No improvement")

    return current_x, current_value

best_x, best_value = hill_climb(start=random.uniform(-10, 10), step_size=0.1, max_iterations=10)

print("\nBest solution found:")
print(f"x = {best_x:.4f}, f(x) = {best_value:.4f}")