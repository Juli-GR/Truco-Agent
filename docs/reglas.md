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

## Etapa 1: envido aislado

- El nodo de azar reparte directamente el par de tantos (t₁, t₂) con la probabilidad exacta que surge
  del reparto de 3 cartas a cada jugador de un mazo de 40.
- Es una abstracción sin pérdida: en el envido aislado la mano solo importa a través de los tantos.
- Si nadie canta, la recompensa es 0.
