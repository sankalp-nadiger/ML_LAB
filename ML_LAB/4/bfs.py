from queue import PriorityQueue

def best_first_search(graph, start, goal, heuristic):
    open_list = PriorityQueue()
    open_list.put((heuristic[start], start))
    visited = set()

    path = []

    while not open_list.empty():
        _, current = open_list.get()
        if current in visited:
            continue

        path.append(current)
        visited.add(current)

        if current == goal:
            return path

        for neighbor, _ in graph[current]:
            if neighbor not in visited:
                open_list.put((heuristic[neighbor], neighbor))

    return None

graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('S', 1), ('C', 3), ('D', 2)],
    'B': [('S', 4), ('D', 5)],
    'C': [('A', 3), ('G', 4)],
    'D': [('A', 2), ('B', 5), ('G', 6)],
    'G': [('C', 4), ('D', 6)]
}

heuristic = {
    'S': 7,
    'A': 6,
    'B': 2,
    'C': 1,
    'D': 3,
    'G': 0
}

path = best_first_search(graph, 'S', 'G', heuristic)
print("Best First Search Path:", path)