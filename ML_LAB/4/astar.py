def a_star(graph, start, goal, h):
    open_list = [start]
    closed_list = set()
    
    g = {start: 0}
    parents = {start: None}
    
    while open_list:
        # Sort by f = g + h
        open_list.sort(key=lambda v: g[v] + h(v))
        n = open_list.pop(0)
        
        if n == goal:
            # Reconstruct path
            path = []
            while n is not None:
                path.append(n)
                n = parents[n]
            path.reverse()
            print("Path found:", path)
            return path
        
        closed_list.add(n)
        
        for (m, weight) in graph[n]:
            if m in closed_list:
                continue
            
            new_g = g[n] + weight
            if m not in open_list or new_g < g.get(m, float('inf')):
                g[m] = new_g
                parents[m] = n
                if m not in open_list:
                    open_list.append(m)
    
    print("Path does not exist")
    return None

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1), ('E', 3)],
    'E': [('D', 3)]
}

h = lambda n: {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 0}[n]

# Run
a_star(graph, 'A', 'E', h)