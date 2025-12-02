import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import heapq

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

# Entrada: Nodos y lista de adyacencia
# Salida: La función en si no retorna nada, solo muestra el gráfico de la red
# Complejidad: O(n^2)
def graficar_network(nodos, adj):
    plt.figure(figsize=(8,8)) # Tamaño de la figura

    # Nodos
    for nid, (x,y,tipo) in nodos.items():
        color = "red"if tipo == 1 else "blue"
        plt.scatter(x,y,c=color,s=30)
        plt.text(x,y,str(nid),fontsize=8)
    
    # Aristas
    drawn = set() # Evitamos dibujar la misma arista dos veces
    for u,vecinos in adj.items():
        x1,y1, _ = nodos[u]
        for v,_ in vecinos:
            if (v,u) in drawn:
                continue
            drawn.add((u,v))
            x2,y2, _ = nodos[v]
            plt.plot([x1,x2],[y1,y2],c="gray", linewidth=1)
    
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Distribución de agua")
    plt.show()

# TODO: Sectorización de la red
# 1. Dividir la red en sectores, cada uno asociado a una fuente
# 2. Los nodos deben quedar asignados a la fuente más cercana de la red (No en distancia eucladiana sino en el grafo)
# 3. Determinar que tuberías son las que se deben cerrar para separar los sectores
# Cosas que mostar:
#       1. Que nodos pertenecen a cada sector
#       2. Que tuberías se deben cerrar
#       3. Gráfico coloreado con sectores y marcando las tuberías que se deben cerrar

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
        for v, l in adj[U]:
            nd = distancia + l
            if nd < best_dist[v]:
                best_dist[v] = nd
                central_point[v] = src
                heapq.heappush(pq, (nd, src, v)) # distancia , nodo actual , nodo padre
    
    sectors = {}
    for node,src in central_point.items():
        sectors.setdefault(src, []).append(node)
    
    return sectors,central_point,best_dist

#def graficar_sectorizacion(nodos, adj, central, closed_pipes):
#    TODO

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
        q.remove(u)
        procesado[u] = 1

        for (v, l) in vecino[u]:
            nueva_dist = distancia[u] + l
            if nueva_dist < distancia[v]:
                distancia[v] = nueva_dist
                predecesor[v] = u

    return distancia, predecesor


def main(): # Main
    files = get_all_txt_files() # Archivos .txt
    for file in files: # Por cada archivo
        nodes, sources, adj, office, new_nodes = ready_up_graph(file) # Datos del grafo
        pipe_lengths = longitud_tuberias(nodes, adj) # Longitudes de las tuberías
        sectors, central, distances = sectorizacion(sources, adj)
        agua_dist, agua_pred = calidad_agua(office, nodes, adj)
        print(f"Archivo: {file}")
        print(f"  Sectores: {sectors}")
        print(f"  Distancias a fuentes: {distances}")
        print(f"  Calidad del agua desde oficina: {agua_dist}")

if __name__ == "__main__":
    main()