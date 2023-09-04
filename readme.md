# do not readme

- es un algoritmo de clusterización que corta progresivamente el conjunto de puntos en subconjuntos, para obtener particiones balanceadas.
- el algoritmo es O(n log^2 n) según el Teorema Maestro.
- el balanceo se logra cortando en lugares precisos para el que estadísticamente debería resultar en subparticiones balanceadas.
- En términos de teoría, se puede utilizar desigualdad de concentración de Azuma-Hoeffding para demostrar que el resultado es balanceado (obteniendose cotas de concentración).

Dudas me pueden preguntar.

## Some pics

<img src="images/normal.png" />
<img src="images/normal2.png" />

<img src="images/uniform.png" />
<img src="images/uniform2.png" />

<img src="images/exponential.png" />
<img src="images/exponential2.png" />

## Some histograms

<img src="images/hist1.png" />
<img src="images/hist2.png" />
<img src="images/hist3.png" />
<img src="images/hist4.png" />

as you can see, *it is balanced*
