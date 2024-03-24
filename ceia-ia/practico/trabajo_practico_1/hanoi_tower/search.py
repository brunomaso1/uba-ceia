from collections import deque
from hanoi_tower import tree_hanoi
from hanoi_tower import hanoi_states


def breadth_first_tree_search(problem: hanoi_states.ProblemHanoi):
    """
    Realiza una búsqueda en anchura para encontrar una solución a un problema de Hanoi.
    Esta función no chequea si un estado se visito, por lo que puede entrar en Loop infinitos muy fácilmente. No
    usarla con más de 3 discos.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """
    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)]
                     )  # Creamos una cola FIFO con el nodo inicial
    while frontier:
        node = frontier.popleft()  # Extraemos el primer nodo de la cola
        # Comprobamos si hemos alcanzado el estado objetivo
        if problem.goal_test(node.state):
            return node
        # Agregamos a la cola todos los nodos sucesores del nodo actual
        frontier.extend(node.expand(problem))

    return None


def breadth_first_graph_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda en anchura para encontrar una solución a un problema de Hanoi. Pero ahora si recuerda si ya
    paso por un estado e ignora seguir buscando en ese nodo para evitar recursividad.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """

    # Creamos una cola FIFO con el nodo inicial
    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)])

    explored = set()  # Este set nos permite ver si ya exploramos un estado para evitar repetir indefinidamente
    while frontier:
        node = frontier.popleft()  # Extraemos el primer nodo de la cola

        # Agregamos nodo al set. Esto evita guardar duplicados, porque set nunca tiene elementos repetidos, esto sirve
        # porque heredamos el método __eq__ en tree_hanoi.NodeHanoi de aima.Node
        explored.add(node.state)

        # Comprobamos si hemos alcanzado el estado objetivo
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "caminos se expandieron y", len(
                    frontier), "caminos quedaron en la frontera")
            return node
        # Agregamos a la cola todos los nodos sucesores del nodo actual que no haya visitados
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)

    return None


def depth_first_graph_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda en profundidad para encontrar una solución a un problema de Hanoi. Recuerda los nodos para evitar la recursividad.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """

    # Creamos una pila LIFO con el nodo inicial
    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)])

    explored = set()  # Este set nos permite ver si ya exploramos un estado para evitar repetir indefinidamente
    while frontier:
        node = frontier.pop()  # Extraemos el último nodo de la pila

        # Agregamos nodo al set. Esto evita guardar duplicados, porque set nunca tiene elementos repetidos, esto sirve
        # porque heredamos el método __eq__ en tree_hanoi.NodeHanoi de aima.Node
        explored.add(node.state)

        # Comprobamos si hemos alcanzado el estado objetivo
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "caminos se expandieron y", len(
                    frontier), "caminos quedaron en la frontera")
            return node
        # Agregamos a la pila todos los nodos sucesores del nodo actual que no hayan sido visitados
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)

    return None


def depth_limited_search(problem: hanoi_states.ProblemHanoi, limit: int = 1, display: bool = False):
    """
    Realiza una búsqueda en profundidad para encontrar una solución a un problema de Hanoi. Recuerda los nodos para evitar la recursividad.
    Está limitada a una profundiad dada.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        limit (int): Limite de profundidad.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """

    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)])

    explored = set()
    while frontier:
        node = frontier.pop()

        explored.add(node.state)

        if problem.goal_test(node.state):
            if display:
                print(len(explored), "caminos se expandieron y", len(
                    frontier), "caminos quedaron en la frontera")
            return node
        # Agregamos a la pila todos los nodos sucesores del nodo actual que no hayan sido visitados, pero también inlcuimos el límite.
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier and child.depth <= limit)

    return None


def depth_iterative_limited_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda en profundidad para encontrar una solución a un problema de Hanoi. Recuerda los nodos para evitar la recursividad.
    Está limitada a una profundiad dada, sin embargo, esta es iterativa hasta infinito, por lo que, sin no hay solución
    el algoritmo no converge.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """

    depth = 1
    while True:
        # Ejecuto la busqueda para un límite dado
        last_node = depth_limited_search(problem, depth, display)
        if last_node != None:  # Si el último nodo no es nulo, entonces es la solución.
            return last_node
        depth += 1

def heuristicFun(actualNode: hanoi_states.StatesHanoi, goal: hanoi_states.StatesHanoi):
    """
    La función heurística es una implementación de la vista en clase:
        h(n) es un punto menos por cada disco ubicado en la posición correcta.
    """
    h_cost = 0
    
    # Al hacer zip(goal.rods, actualNode.rods) se obtiene un iterador, en donde goalSubList y actualNodeSubLis son listas, de la forma [[5, 4], [3, 2], [1]]
    # Al hacer nuevamente zip, se obtiene los elementos de dichas listas.
    # Comparando los elementos, podemos agregar el costo para el caso que sean iguales.
    for goalSubList, actualNodeSubLis in zip(goal.rods, actualNode.rods):
        for goalSubListElement, actualNodeSubLisElement in zip(goalSubList, actualNodeSubLis):
            h_cost += -1 if goalSubListElement == actualNodeSubLisElement else 0
    return h_cost
    
def greedy_best_first_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda voraz, utilizando una funcion heurística.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """
    
    def heuristicFunWrapper(node: tree_hanoi.NodeHanoi, problem: hanoi_states.ProblemHanoi = problem):
        return heuristicFun(node.state, problem.goal)

    # Dado que los elementos los vamos insertar ordenados, creamos una lista FIFO con el nodo inicial.
    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)])

    explored = set()
    while frontier:
        node = frontier.pop()

        explored.add(node.state)

        if problem.goal_test(node.state):
            if display:
                print(len(explored), "caminos se expandieron y", len(frontier), "caminos quedaron en la frontera")
            return node
        
        # Insertamos los elementos ordenados según la función heurística.
        frontier.extend([child for child in sorted(node.expand(problem), key=heuristicFunWrapper, reverse=True)
                        if child.state not in explored and child not in frontier])
    return None

def a_star_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda A*, utilizando una función heurística.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """
    
    def totalCostPlusHeuristic(node: tree_hanoi.NodeHanoi, problem: hanoi_states.ProblemHanoi = problem):
        return heuristicFun(node.state, problem.goal) + node.path_cost

    # Dado que los elementos los vamos insertar ordenados, creamos una lista FIFO con el nodo inicial.
    frontier = deque([tree_hanoi.NodeHanoi(problem.initial)])

    explored = set()
    while frontier:
        node = frontier.pop()

        explored.add(node.state)

        if problem.goal_test(node.state):
            if display:
                print(len(explored), "caminos se expandieron y", len(frontier), "caminos quedaron en la frontera")
            return node
        
        # Insertamos los elementos ordenados según la función heurística.
        frontier.extend([child for child in sorted(node.expand(problem), key=totalCostPlusHeuristic, reverse=True)
                        if child.state not in explored and child not in frontier])
    return None