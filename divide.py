import numpy as np
from typing import Callable,List
from matplotlib import pyplot as plt


## Implementation ##################################################################################

# un punto está definido como (x,y,w)
# donde x,y son sus coordenadas en R2 y w es un peso
# el algoritmo de clustering es de tipo jerárquico,
# y busca dividir el conjunto de puntos en pedazos pequeños y balanceados en peso

class treeNode:
    pass

class treeNode:
    # members (non-static)
    # children : List[treeNode] ; elements : List[np.ndarray]
    # commodity_weight    : float
    sub_clustering_algorithm : Callable # treeNode -> treeNode

    #  " constructor ", llama este método con una lista de puntos [x,y,w]
    @staticmethod
    def make(input_data : List[np.ndarray]):
        commodities = sum( (x[2] for x in input_data ) )
        new = treeNode([],input_data,commodities)
        new.run()
        return new

    # methods
    def is_leaf(self):
        return len(self.children)==0
    
    def is_internal(self):
        return not self.is_leaf
    
    def __init__(self,
                 children : List[treeNode],
                 elements : List[np.ndarray],
                 commodity_weight : float ,
                 ):
        self.children = children
        self.elements = elements
        self.commodity_weight = commodity_weight
    
    def run(self):
        # do the thing
        # first, separate the node into children
        result = treeNode.sub_clustering_algorithm(self)
        self.children = result.children
        self.elements = result.elements
        # recursive call
        for child in self.children:
            child.run()
        
    def get_clusters(self):
        C = []
        # should recursively append C with root elements
        treeNode.get_process_branch(self,C)
        return C
    
    @staticmethod
    def get_process_branch(t : treeNode, L : list):
        if t.is_leaf():
            L.append(t.elements)
            return
        # else, it is internal
        for x in t.children:
            treeNode.get_process_branch(x,L)

lower_thresh = 200

## sub-clustering procedure ########################################################################

# si el cluster es chiquito, lo deja tranquilo
# si no,
#   elige una recta al azar en R2
#   mueve la recta hasta que divida la mitad del cluster, en términos de peso
#   crea dos clusteres a partir de esta división
def randomSplit(n : treeNode, K : int = 2):
    global lower_thresh
    if (n.commodity_weight<lower_thresh): return n
    # else

    # calculate most  disperse direction
    var = np.var(n.elements,axis=0)
    coordinate = lambda x : x[0] - np.sin(x[1])*0.2
    if var[1] > var[0]: coordinate = lambda x : x[1] - np.cos(x[0])*0.2 
    
    # ordenar los elementos según la coordenada y calcular el peso debajo recta
    # de la recta que pasa por cada punto
    arr = np.array(sorted([[coordinate(x),x[2]] for x in n.elements]))
    arr_sum = arr[:,1].cumsum()

    # encontrar los coeficientes z_k para que ax+by+z divida el cluster justo en 1/K del peso
    I = [ np.searchsorted(arr_sum, (n.commodity_weight * float(k))/K) for k in range(K) ]
    for j,i in enumerate(I):
        if i<0:
            I[j] = 0
        if i>=len(arr):
            I[j] = len(arr)-1

    Z = [arr[i][0] for i in I]; Z.append(np.inf)
    clusters = [
        [x for x in n.elements if (low <= coordinate(x) < high)] for low, high in zip(Z, Z[1:])
    ]
    new_nodes = [treeNode([],c,sum(x[2] for x in c)) for c in clusters]

    ## este nodo reemplaza el nodo de entrada
    res = treeNode(new_nodes,[],n.commodity_weight)
    return res 

treeNode.sub_clustering_algorithm = lambda x : randomSplit(x,2)

# ejemplo
lower_thresh = 300
def xy():
    x = np.random.normal()
    y = np.random.exponential()
    return (x,y)

X = lambda : np.random.uniform(low=1,high=4)

M = 100000 # ufa, esto es grande
input_data = [np.array([*xy(),X()]) for _i in range(M)]

print("comenzando!")
T = treeNode.make(input_data)
K = T.get_clusters()
print("listo!")

plt.style.use('dark_background')
vals = [sum(x[2] for x in k) for k in K]
plt.hist(vals,20)

fig,ax = plt.subplots(figsize=(10,10))

for k in K:
    xx,yy,__ = zip(*k)
    ax.scatter(xx,yy,s=8.,marker="D")

plt.show()

## recta al azar
#a = np.random.uniform(-1,1)
#b = np.random.uniform(-1,1)
#coordinate = lambda x : a*x[0] + b*x[1]