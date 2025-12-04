from pathlib import Path
import math
import matplotlib.pyplot as plt
import heapq
from collections import deque   

# Términos importantes
# Lista de adyacencia: Lista que representa las conexiones entre nodos en un grafo.

# Entrada: ninguno, usa el directorio actual
# Salida: lista de archivos .txt en el directorio actual
# Complejidad: O(n)
def get_all_txt_files():
    directory = Path("./")
    txt_files = list(directory.glob("*.txt"))
    return txt_files
 
# Entrada: nombre del archivo de texto con la descripción del grafo
# Salida: diccionario de nodos, conjunto de fuentes, lista de adyacencia, oficina y nuevos nodos
# Complejidad: O(n)
def ready_up_graph(filename: str):
    nodes = {}
    sources = set()
    adj = {}

    with open(filename, "r") as file:
        n_nodes, n_edges = map(int, file.readline().split())

        line = file.readline().strip()
        while line != "[NODES]":
            line = file.readline().strip()

        for line in file:
            line = line.strip()
            if line == "[EDGES]":
                break
            if not line:
                continue
            node_id, x, y, tipo = line.split()
            node_id = int(node_id)
            x, y = float(x), float(y)
            tipo = int(tipo)
            nodes[node_id] = (x, y, tipo)
            if tipo == 1:
                sources.add(node_id)
            adj.setdefault(node_id, [])

        for line in file:
            line = line.strip()
            if line == "[OFFICE]":
                break
            if not line:
                continue
            u, v, l = line.split()
            u, v = int(u), int(v)
            l = float(l)
            adj.setdefault(u, []).append((v, l))
            adj.setdefault(v, []).append((u, l))  # grafo no dirigido

        office = int(file.readline().strip())

        for line in file:
            line = line.strip()
            if line == "[NEW]":
                break

        new_nodes = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            x, y, d = line.split()
            new_nodes.append((float(x), float(y), float(d)))

    return nodes, sources, adj, office, new_nodes
 
# Entrada: diccionario de nodos y lista de adyacencia del grafo
# Salida: lista de tuplas con pares de nodos, longitud declarada y longitud geométrica
# Complejidad: O(n)
def longitud_tuberias(nodos, adj):
    pipe_lengths = [] # Lista de resultados
    seen = set() # Cuales ya visitamos
    for u, vecinos in adj.items(): # Escogemos un nodo y sus vecinos
        for v, l in vecinos: # Escogemos un vecino y sacamos la longitud con respecto al nodo
            if (v, u) in seen: # Si ya lo hemos visitado, lo ignoramos
                continue
            seen.add((u, v)) # Agregamos a visitados
            x1, y1, _ = nodos[u] # Primer nodo
            x2, y2, _ = nodos[v] # Segundo nodo
            temp = math.hypot(x2 - x1, y2 - y1) # Hipotenusa de las coordenadas , o en otras palabras la longitud
            pipe_lengths.append((u, v, l, temp)) # Agregamos el resultado a la lista
    return pipe_lengths

# Graficar la red con sectores, tuberías cerradas, oficina y ruta
def graficar_network(nodos, adj, central=None, closed_pipes=None, office=None, ruta_calidad=None, nuevos_nodos=None):
    plt.figure(figsize=(14,10))

    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    
    drawn = set()
    for u, vecinos in adj.items():
        if u not in nodos:
            continue
        x1, y1, _ = nodos[u]
        for v, _ in vecinos:
            if v not in nodos or (v, u) in drawn:
                continue
            drawn.add((u, v))
            x2, y2, _ = nodos[v]
            
            cerrada = closed_pipes and ((u, v) in closed_pipes or (v, u) in closed_pipes)
            if cerrada:
                plt.plot([x1, x2], [y1, y2], c="red", linewidth=3, linestyle='--', 
                        label='Tubería cerrada' if (u,v) == closed_pipes[0] else '', zorder=1)
            else:
                plt.plot([x1, x2], [y1, y2], c="gray", linewidth=1, alpha=0.5, zorder=1)
    
    if ruta_calidad and len(ruta_calidad) > 1:
        for i in range(len(ruta_calidad) - 1):
            u, v = ruta_calidad[i], ruta_calidad[i+1]
            if u in nodos and v in nodos:
                x1, y1, _ = nodos[u]
                x2, y2, _ = nodos[v]
                plt.plot([x1, x2], [y1, y2], c="green", linewidth=2.5, alpha=0.7, 
                        label='Ruta de calidad' if i == 0 else '', zorder=2)
    
    for nid, (x, y, tipo) in nodos.items():
        if central and nid in central:
            sid = central[nid]
            slist = sorted(set(central.values()))
            if sid in slist:
                idx = slist.index(sid) % len(colores)
                color = colores[idx]
            else:
                color = "gray"
        else:
            color = "red" if tipo == 1 else "blue"
        
        size = 100 if tipo == 1 else 50
        
        if office and nid == office:
            plt.scatter(x, y, c="gold", s=150, marker='s', edgecolors='black', 
                       linewidths=2, zorder=4, label='Oficina')
        elif tipo == 1:
            plt.scatter(x, y, c=color, s=size, marker='^', edgecolors='black', 
                       linewidths=2, zorder=3, 
                       label='Fuente' if nid == min([n for n in nodos if nodos[n][2] == 1]) else '')
        else:
            plt.scatter(x, y, c=color, s=size, edgecolors='black', linewidths=0.5, zorder=3)
        
        plt.text(x, y+0.3, str(nid), fontsize=8, ha='center', fontweight='bold')
    
    if nuevos_nodos:
        for i, (x, y, _) in enumerate(nuevos_nodos):
            plt.scatter(x, y, c="purple", s=150, marker='*', edgecolors='black', 
                       linewidths=2, zorder=5, label='Nuevo nodo' if i == 0 else '')
            plt.text(x, y+0.4, f"N{i+1}", fontsize=9, ha='center', 
                    fontweight='bold', color='purple')
    
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Red de Distribución de Agua")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

# TODO: Sectorización de la red
# 1. Dividir la red en sectores, cada uno asociado a una fuente
# 2. Los nodos deben quedar asignados a la fuente más cercana de la red (No en distancia eucladiana sino en el grafo)
# 3. Determinar que tuberías son las que se deben cerrar para separar los sectores
# Cosas que mostar:
#       1. Que nodos pertenecen a cada sector
#       2. Que tuberías se deben cerrar
#       3. Gráfico coloreado con sectores y marcando las tuberías que se deben cerrar


# Entrada: central y lista de adyacencia
# Salida: lista de tuberías que se deben cerrar
# Complejidad: O(n^2)
def establecer_closed_pipes(central, adj):
    closed = []
    seen = set()
    for u, n in adj.items():
        for v, _ in n:
            if (u,v) in seen or (v,u) in seen:
                continue
            seen.add((u,v))
            if central[u] != central[v]:
                closed.append((u,v))
    return closed


# Entrada: fuentes y lista de adyacencia
# Salida: sectores, central_point y best_dist
# Complejidad: O(n^2)
def sectorizacion(sources, adj):
    inf_float = float("inf")
    best_dist = {u: inf_float for u in adj} # Distancia mínima a una fuente
    central_point = {u: None for u in adj} # Punto central de la fuente

    pq = []
    for i in sources:
        best_dist[i] = 0.0
        central_point[i] = i
        heapq.heappush(pq, (0.0,i,i)) # distancia , nodo actual , nodo padre
    
    while pq:
        distancia, src, u = heapq.heappop(pq)
        if distancia != best_dist[u] or central_point[u] != src:
            continue # Si la distancia no es la mejor o no es el nodo padre, ignoramos
        for v, l in adj[u]:
            nd = distancia + l
            if nd < best_dist[v]:
                best_dist[v] = nd
                central_point[v] = src
                heapq.heappush(pq, (nd, src, v)) # distancia , nodo actual , nodo padre
    
    sectors = {}
    for node,src in central_point.items():
        sectors.setdefault(src, []).append(node)
    
    return sectors,central_point,best_dist

# Dijkstra para calcular distancias desde un nodo origen
def calidad_agua(origen, nodos, vecino):
    distancia = {n: float('inf') for n in nodos}
    procesado = {n: 0 for n in nodos}
    predecesor = {n: None for n in nodos}

    distancia[origen] = 0
    q = [n for n in nodos]
    while q:
        u = None
        menor = float('inf')
        for n in q:
            if distancia[n] < menor:
                menor = distancia[n]
                u = n
        if u is None:
            break
        q.remove(u)
        procesado[u] = 1

        for (v, l) in vecino[u]:
            nd = distancia[u] + l
            if nd < distancia[v]:
                distancia[v] = nd
                predecesor[v] = u

    return distancia, predecesor

# TODO: Usar las capacidades de las tuberías para calcular el flujo máximo en cada sector
# Considerar: 
# 1. origen = la fuente del sector
# 2. Destino = el nodo más alejado de la fuente en ese sector
# Podemos usar el algoritmo de Edmonds-Karp o Ford-Fulkerson

# Entrada: grafo, origen y destino
# Salida: flujo máximo
# Complejidad: O(n^2)
def ed_karp(c,s,t):
    # Si destino y origen es el mismo , pasé 2 horas debugeando , escribir esto sabe a gloria
    if s == t:
        return 0.0
    
    n = len(c) # Longitud de nodos 
    max_flow = 0 # Resultado
    iteration = 0
    MAX_ITERATIONS = 1000  # Protección contra bucles infinitos

    # Obtener todos los nodos (claves y valores)
    all_nodes = set(c.keys())
    for u in c:
        all_nodes.update(c[u].keys())

    while True:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            print(f"ADVERTENCIA: Edmonds-Karp alcanzó {MAX_ITERATIONS} iteraciones. Posible bucle infinito.")
            break
        parent = {u: None for u in all_nodes} # Para cada nodo, su padre
        parent[s] = s # Padre
        q = deque([s]) # Cola de nodos

        while q and parent[t] is None: # Mientras haya camino
            u = q.popleft() # Utilizamos el primer elemento
            if u not in c:  # Si el nodo no tiene vecinos, continuar
                continue
            for v, cap in c[u].items(): # Recorremos los vecinos
                if cap > 1e-9 and v in parent and parent[v] is None: # Si hay capacidad y no hemos visitado
                    parent[v] = u # Padre
                    q.append(v) # Agregamos al final
        
        if parent[t] is None: # No hay camino
            break

        # Flujo máximo del camino
        bottleneck = float('inf') # EL bottleneck es cuando ya hay un tope en el camino
        v = t # El destino
        while v != s: # Mientras no lleguemos al origen
            u = parent[v] # El padre
            bottleneck = min(bottleneck, c[u][v]) # Mínima capacidad en el camino
            v = u
        
        v = t
        while v != s:
            u = parent[v]
            c[u][v] -= bottleneck # Restamos el bottleneck
            c[v].setdefault(u,0.0)
            c[v][u] += bottleneck # Sumamos el bottleneck
            v = u
        
        max_flow += bottleneck
    return max_flow


# Entrada: nodos y lista de adyacencia
# Salida: nodo más alejado
# Complejidad: O(n)
def find_furthest_node(nodos, mejor_dist):
    furthest_node = None
    max_distance = float('-inf')
    for node in nodos:
        if mejor_dist[node] > max_distance:
            max_distance = mejor_dist[node]
            furthest_node = node
    return furthest_node

# Entrada: sectores, mejor_dist y lista de adyacencia
# Salida: flujo máximo en cada sector
# Complejidad: O(n^2)
def flujo_max_sector(sectores, mejor_dist, adj):
    resultados = {} # fuente - destino_mas_alejado - max_flow

    # Preparamos el grafo
    for fuente,nodos in sectores.items():
        if not nodos:  # sector vacío, por seguridad
            continue

        # Si el sector tiene un solo nodo, no hay flujo que calcular
        if len(nodos) == 1:
            resultados[fuente] = (fuente, 0.0)
            continue

        furthest_node = find_furthest_node(nodos, mejor_dist)
        if furthest_node is None or furthest_node == fuente:  # por seguridad
            resultados[fuente] = (fuente, 0.0)
            continue

        n_set = set(nodos)
        cap = {u: {} for u in n_set}  # diccionario de capacidades

        # Usar la longitud como capacidad (o un valor fijo si prefieres)
        # Nota: Si las tuberías tienen capacidad real, debes modificar esto
        for u in n_set:
            if u not in adj:
                continue
            for v, longitud in adj[u]:
                if v in n_set:
                    # Usar un valor de capacidad basado en la longitud
                    # Capacidad inversamente proporcional a la longitud
                    # o usar un valor fijo como 100.0
                    capacidad = 100.0  # Valor fijo de capacidad
                    cap[u].setdefault(v, 0.0)
                    cap[u][v] = max(cap[u][v], capacidad)

        # Ed Karp
        max_flow = ed_karp(cap, fuente, furthest_node)
        resultados[fuente] = (furthest_node, max_flow)

    return resultados

# Ruta de muestreo de calidad del agua
# TODO:
# Suponer que una persona inicia en el nodo *office*.
# Determinar una ruta de distancia mínima que:
#   - Visite todos los nodos
#   - Regrese al nodo office al final.
# Es un problema tipo TSP sobre el grafo, asumiendo que las calles sigan las tuberías.
# Mejor implementación: Maybe búsqueda exhaustiva de tipo greedy o un Minimo Spanning Tree.

# dist_amtriz: matriz de distancias más cortas entre todos los pares de nodos

# Entrada: grafo de adyacencia
# Salida: matriz de distancias más cortas entre todos los pares de nodos
# Complejidad: O(n^3)
def floyd_warshall(adj):
    # Distancias más cortas entre dos puntos
    nodes = list(adj.keys())
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    
    # Distancia de un nodo a sí mismo es 0, no vuelvo a caer
    for u in nodes:
        dist[u][u] = 0.0
    
    # Distancias directas del grafo
    for u in adj:
        for v, l in adj[u]:
            dist[u][v] = min(dist[u][v], l)
    
    # Floyd-Warshall
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

# Entrada: ruta (lista de nodos), matriz de distancias
# Salida: distancia total de la ruta
# Complejidad: O(n)
def office_a_office(r, dist_matrix):
    # Calcula la distancia total de una ruta usando la matriz de distancias más cortas
    t = 0.0
    for i in range(len(r) - 1):
        u, v = r[i], r[i+1]
        if v in dist_matrix.get(u, {}):
            t += dist_matrix[u][v]
        else:
            # Si no hay camino, retornar infinito
            return float('inf')
    return t


# Entrada: nodo inicial (o), grafo (adj), matriz de distancias
# Salida: ruta completa que visita todos los nodos y regresa a o
# Complejidad: O(n^2)
def calidad_busqueda_exhaustiva(o, adj, dist_matrix):
    visited = set([o])
    ruta = [o]
    current = o
    nodes = list(adj.keys())

    # Nearest Neighbor: mientras haya nodos sin visitar
    while len(visited) < len(nodes):
        m_v = None
        m_l = float('inf')
        
        # Buscar el nodo no visitado más cercano (usando distancia más corta)
        for v in nodes:
            if v in visited:
                continue
            # Solo considerar nodos que están conectados
            if current in dist_matrix and v in dist_matrix[current]:
                dist = dist_matrix[current][v]
                if dist < m_l:
                    m_l = dist
                    m_v = v
        
        if m_v is None:
            # Si no hay más nodos alcanzables, el grafo no es conexo
            break
        
        visited.add(m_v)
        ruta.append(m_v)
        current = m_v
    
    ruta.append(o)  # Regresamos al nodo inicial
    return ruta


# Agregar nuevos nodos a la red
def expandir_red(nodos, adj, sources, new_nodes):
    next_id = max(nodos.keys()) + 1 if nodos else 1
    nuevos_agregados = []
    
    for x_new, y_new, d_max in new_nodes:
        nodo_cercano = None
        dist_minima = float('inf')
        
        for nid, (x, y, tipo) in nodos.items():
            if tipo == 1:
                continue
            dist = math.hypot(x_new - x, y_new - y)
            if dist <= d_max and dist < dist_minima:
                dist_minima = dist
                nodo_cercano = nid
        
        if nodo_cercano is None:
            for nid, (x, y, tipo) in nodos.items():
                dist = math.hypot(x_new - x, y_new - y)
                if dist <= d_max and dist < dist_minima:
                    dist_minima = dist
                    nodo_cercano = nid
        
        if nodo_cercano is not None:
            nodos[next_id] = (x_new, y_new, 0)
            adj.setdefault(next_id, [])
            
            adj[nodo_cercano].append((next_id, dist_minima))
            adj[next_id].append((nodo_cercano, dist_minima))
            
            nuevos_agregados.append((next_id, nodo_cercano, dist_minima))
            next_id += 1
        else:
            print(f"ADVERTENCIA: No se pudo conectar el nuevo nodo en ({x_new}, {y_new}) - fuera de rango")
    
    return nodos, adj, nuevos_agregados

def main():
    files = get_all_txt_files()
    for file in files:
        nodes, sources, adj, office, new_nodes = ready_up_graph(file)
        pipe_lengths = longitud_tuberias(nodes, adj)
        sectors, central, distances = sectorizacion(sources, adj)
        agua_dist, agua_pred = calidad_agua(office, nodes, adj)
        flujo_max = flujo_max_sector(sectors, distances, adj)
        
        dist_matrix = floyd_warshall(adj)
        ruta_calidad = calidad_busqueda_exhaustiva(office, adj, dist_matrix)
        distancia_ruta = office_a_office(ruta_calidad, dist_matrix)

        print("--------------------------------")
        print("Resultados obtenidos hasta el momento")
        print("Archivo: ", file)
        print("Longitudes de las tuberías: ", pipe_lengths)
        print("Calidad del agua: ", agua_dist)
        print("Flujo máximo en cada sector: ", flujo_max)
        print("Ruta de calidad del agua: ", ruta_calidad)
        print("Distancia total de la ruta: ", distancia_ruta)
        print("--------------------------------\n\n")
        
        # Graficar la red completa
        closed_pipes = establecer_closed_pipes(central, adj)
        graficar_network(nodes, adj, central, closed_pipes, office, ruta_calidad, None)



if __name__ == "__main__":
    main()