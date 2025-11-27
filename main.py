import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
 
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

def graficar_network(nodos, adj):
    plt.figure(figsize=(8,8))

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

def main(): # Main
    files = get_all_txt_files() # Archivos .txt
    for file in files: # Por cada archivo
        nodes, sources, adj, office, new_nodes = ready_up_graph(file) # Datos del grafo
        pipe_lengths = longitud_tuberias(nodes, adj) # Longitudes de las tuberías
        graficar_network(nodes, adj)
if __name__ == "__main__":
    main()