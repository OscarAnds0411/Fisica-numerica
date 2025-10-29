""" "
Este problema pretende estudiar las
ocscilaciones de una cuerda. Considere una cuerda de longitud L y
densidad  (x) por unidad de longitud, atada en ambos extremos y
bajo una tensiÛn T (x). Suponga que el desplazamiento relativo de la
cuerda respecto a su posiciÛn de equilibrio y(x;t)
L
es pequeÒo y que la

pendiente de la cuerda @y

@x tambien es pequeÒa.

(a) Considere una secciÛn inÖnitesimal de la cuerda, como se muestra

en la Ögura, note que la diferencia en las componentes de las ten-
siones en x y x + x tiene por resultado una fuerza restauradora.

Demuestre que al aplicar las leyes de Newton a esta secciÛn obten-
emos la ecuaciÛn de onda

dT (x)
dx
@y (x; t)
@x + T (x)
@
2
y (x; t)
@x2
=  (x)
@
2
y (x; t)
@t2 (b) øQuÈ condiciones son necesarias para obtener la ecuaciÛn de onda
est·ndar
@
2
y (x; t)
@x2
=
1
c
2
@
2
y (x; t)
@t2
; c =
s
T

?

(c) øQuÈ condiciones deben cumplirse para obtener una  ̇nica soluciÛn
a esta EDP de segundo orden?
(d) Utilice una malla de pasos de longitud t en el tiempo y x en
el espacio para obtener una soluciÛn numÈrica
y (x; t) = y (ix; jt) = yi;j :

(e) Exprese las segundas derivadas de la EDP en tÈrminos de difer-
encias Önitas, demuestre que esto resulta en la ecuaciÛn de onda

en diferencias:
yi;j+1 + yi;j
1
2yi;j
c
2
(t)
2 =

yi+1;j + yi
1;j
2yi;j
(x)
2
:
(f) Demuestre que el algoritmo anterior puede escribirse como

yi;j+1 = 2yi;j
yi;j
1 +
c
2
c
02
[yi+1;j + yi
1;j
2yi;j ] ;

donde c
0 =
x
t
es la velocidad de la malla, es decir, la razÛn

numÈrica de los par·metros.

(g) øCÛmo entran las condiciones iniciales y las condiciones de fron-
tera?

(h) La condiciÛn de Courant para la estabilidad de la soluciÛn es que

c
c
0
 1:
øquÈ signiÖca en tÈrminos de los pasos?

3

(i) Escriba un programa que implemente la fÛrmula anterior, pro-
duzca una animaciÛn con el movimiento de la cuerda.

(j) Cambie los pasos del tiempo y el espacio en su programa para
que algunas veces satisfaga la condiciÛn de Courant y otras no.
Describa quÈ pasa en cada caso.

Cuando la condiciÛn de Courant se satisface, la simulaciÛn se comporta de manera estable y las ondas en la cuerda se propagan correctamente. Sin embargo, si la condiciÛn de Courant no se satisface, la simulaciÛn puede volverse inestable, lo que resulta en oscilaciones no físicas y un comportamiento errático de la cuerda.


"""
