# Reglas implementadas

Truco argentino 1v1, sin flor. Este documento fija las reglas exactas que implementa el código.

## Generales

- Mazo español de 40 cartas (sin 8 ni 9).
- El mano se elige al azar al inicio de cada mano (nodo de azar), por lo que el juego es simétrico.
- La Falta Envido queda excluida en todas las etapas sin marcador (1 a 3).

## Envido

### Tantos

- Con dos o más cartas del mismo palo: 20 + la suma de las dos más altas de ese palo.
- Sin dos cartas del mismo palo: el valor de la carta más alta.
- Las figuras (10, 11, 12) valen 0; las demás cartas valen su número.
- Rango posible: 0 a 33.

### Cantos

- Envido (2 puntos): se puede cantar hasta dos veces.
- Real Envido (3 puntos): una sola vez, y después de él no se puede subir más.
- Secuencias posibles: `E`, `R`, `E-E`, `E-R`, `E-E-R`.
- Ante un canto, el rival responde: quiero, no quiero o sube (si es legal).

| Secuencia | Quiero | No quiero |
|-----------|-------:|----------:|
| E         | 2      | 1         |
| R         | 3      | 1         |
| E-E       | 4      | 2         |
| E-R       | 5      | 2         |
| E-E-R     | 7      | 4         |

### Orden

- El mano puede cantar primero. Si pasa, puede cantar el pie.
- Si nadie canta, el envido no se juega.

### Resolución de un envido aceptado

- El mano declara sus tantos (información pública).
- El pie declara los suyos solo si son mayores; si no, dice "son buenas" y sus tantos quedan ocultos.
- Empate de tantos: gana el mano.

## Truco

### Jerarquía de cartas

De mayor a menor:

1. 1 de espada
2. 1 de basto
3. 7 de espada
4. 7 de oro
5. 3
6. 2
7. 1 de oro y 1 de copa
8. 12
9. 11
10. 10
11. 7 de basto y 7 de copa
12. 6
13. 5
14. 4

Dos cartas del mismo nivel empatan (parda).

### Bazas

- Cada jugador tira una carta por baza; la gana la carta más alta.
- La primera baza la abre el mano. Las siguientes las abre quien ganó la anterior, o el mano si fue
  parda.
- Gana la mano quien gane dos bazas.
- Pardas:
  - Si alguien ganó la primera y la segunda o la tercera sale parda, gana quien ganó la primera.
  - Si la primera es parda, gana quien gane la segunda. Si la segunda también es parda, gana quien gane
    la tercera.
  - Si las tres son pardas, gana el mano.
- La mano termina en cuanto el ganador queda determinado; no se tiran las cartas restantes.

### Cantos

- Sin cantos, la mano vale 1 punto.
- Truco (2 puntos), retruco (3) y vale cuatro (4), en ese orden.
- Ante un canto, el rival responde: quiero, no quiero o sube (si hay un nivel siguiente).
- No quiero: quien cantó último gana el nivel anterior (1, 2 o 3 puntos).
- El truco lo puede cantar cualquiera. Cada subida solo la puede cantar quien aceptó el último canto
  (quien "tiene el quiero"); nadie sube su propio canto.
- Ventana de cantos: antes de cada carta tirada hay una ventana en la que cada jugador tiene una
  oportunidad de cantar, irse al mazo o pasar. Primero actúa quien tiró la última carta; antes de la
  primera carta de la mano, primero actúa el mano.
- Si un canto se resuelve con quiero, la ventana sigue con la oportunidad del otro jugador (si no la
  tuvo todavía). Al cerrar la ventana, tira su carta quien corresponda.

### Irse al mazo

- Un jugador puede irse al mazo en su oportunidad dentro de una ventana de cantos, si no hay un canto
  pendiente de respuesta (con un canto pendiente, no quiero es equivalente).
- El rival gana lo que está en juego: 1 punto si no hay truco aceptado, o el valor del último canto
  aceptado.

## Etapa 1: envido aislado

- El nodo de azar reparte directamente el par de tantos (t₁, t₂) con la probabilidad exacta que surge
  del reparto de 3 cartas a cada jugador de un mazo de 40.
- Es una abstracción sin pérdida: en el envido aislado la mano solo importa a través de los tantos.
- Si nadie canta, la recompensa es 0.

## Etapa 2: truco aislado

Una mano de truco 1v1

## Etapa 3: mano completa con envido

Una mano con envido y truco, con las reglas de arriba más las siguientes.

### Reparto

- El azar reparte cartas reales (con palo), 3 a cada jugador: el palo importa para los tantos.

### Cuándo se puede cantar envido

- El envido se juega una sola vez por mano.
- Un jugador puede cantar envido o real envido en su oportunidad dentro de una ventana de cantos si:
  - todavía no tiró su primera carta,
  - no se jugó el envido en esta mano,
  - y no hay un truco aceptado.
- Cantar envido no usa la oportunidad en la ventana: cuando el envido se resuelve, el mismo jugador
  sigue con su oportunidad (pasar, cantar truco o irse al mazo).
- El envido está primero: ante un truco pendiente, quien responde puede cantar envido o real envido
  en lugar de responder, si cumple las condiciones de arriba. Se resuelve el envido y después
  responde el truco (quiero, no quiero o retruco).
- Una vez cantado el envido, las subidas y respuestas son las del envido aislado.

### Resolución

- Envido no querido: quien cantó último suma los puntos de la tabla y la mano sigue.
- Envido querido: se declaran los tantos como en la etapa 1 y la mano sigue. Los tantos declarados
  (o "son buenas") son información pública para el resto de la mano.
- Al terminar la mano, cada jugador suma los puntos del envido más los del truco. La recompensa va de
  -11 a 11.
- Irse al mazo da solo los puntos del truco; el envido ya jugado se cobra igual.
