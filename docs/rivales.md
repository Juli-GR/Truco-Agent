# Rivales fijos del truco

Con el mazo completo no se puede calcular la exploitabilidad (el árbol tiene del orden de 10¹⁰
nodos), así que los agentes que aprenden se miden jugando contra rivales fijos. Este documento
describe cómo juegan; el código está en `truco/agents/truco_rivales.py`.

| Rival | Idea |
|---|---|
| `aleatorio` | Elige uniformemente entre las acciones legales. Es el piso: cualquier agente debería ganarle. |
| `conservador` | Canta solo con mucha ventaja, nunca farolea y exige margen para aceptar. |
| `equilibrado` | Umbrales intermedios y faroles ocasionales. |
| `agresivo` | Canta con menos ventaja, farolea seguido y le cree poco al rival. |

Los tres heurísticos comparten la misma lógica (cuatro piezas, abajo) y solo cambian sus
parámetros. Juegan como una persona razonable: deciden por **cuánto creen que ganan** según lo que
ya se vio en la mesa, no por la fuerza fija de su mano. Usan solo información que tendría el
jugador: sus cartas, las cartas tiradas y los cantos.

## 1. Juego de cartas

La carta a tirar es determinística (niveles de la jerarquía del truco; "más alta" = mejor carta).

**Si abre la baza:**

- Primera baza: tira una carta media, para no gastar la mejor ni regalar la baza.
- Ya ganó una baza: tira la más baja y guarda la buena.
- Después de una parda: tira la más alta, porque esa baza define.

**Si responde a la carta del rival**, en este orden:

1. Si alguna carta gana la mano en ese momento, tira la más baja de esas (incluye pardar después de
   haber ganado la primera, que también gana la mano).
2. En la primera baza, si puede pardar y le queda una carta buena (que le gane al menos al 60 % del
   mazo), parda: la segunda baza define y la abre él con esa carta.
3. Si alguna carta gana la baza, tira la más baja de esas ("matar con lo justo").
4. Si no puede ganar, tira la más baja.

## 2. Equity: probabilidad de ganar la mano

En cada decisión de canto, el rival calcula su equity *e*:

1. Enumera todas las manos posibles del oponente compatibles con lo visible: sus propias cartas y
   las que el oponente ya tiró.
2. Pondera cada mano por su probabilidad de reparto, dada la mano propia.
3. Para cada una, juega el resto de la mano con las reglas de cartas de la sección 1 (para los dos
   jugadores, sin más cantos) y anota quién gana.
4. *e* es la suma ponderada de las manos en las que gana.

Así la equity se actualiza sola a medida que avanza la mano: después de ganar la primera baza
con un 3 guardado, *e* sube mucho.

## 3. Lectura del rival

Quien canta o acepta suele tener cartas. Por cada canto o "quiero" del oponente, el peso de cada
mano posible se multiplica por

$$\text{percentil}(\text{mano})^{\,\text{lectura}}$$

donde el percentil es la fracción de manos repartidas que son más débiles o iguales (la fuerza de
una mano es el promedio, sobre sus cartas, de la fracción del mazo a la que le gana cada una). Con
`lectura = 0` no le cree nada al rival; con `lectura = 1`, una mano en el percentil 90 pesa nueve
veces más que una en el percentil 10.

## 4. Cantos

**Responder a un canto que vale *v* puntos.** No querer pierde *v* − 1 puntos; querer gana o pierde
*v* (sin contar subidas posteriores). Conviene querer si

$$v(2e - 1) > -(v - 1) \iff e > \frac{1}{2v}$$

o sea, *e* > 25 % ante un truco, *e* > 17 % ante un retruco y *e* > 12,5 % ante un vale cuatro, más
el parámetro `margen`. Antes de eso, si puede subir, sube con la misma regla que para cantar.

**Cantar (o subir).** Canta si *e* supera el umbral del canto (truco, retruco o vale cuatro). Si no
lo supera, farolea con probabilidad `farol`, o el doble si ya ganó una baza. Por los faroles la
política es aleatoria, lo que la hace más difícil de explotar.

**Mazo.** Nunca se va al mazo. En una mano suelta irse al mazo nunca es mejor que jugarla: se
pierden los mismos puntos y no hay manos futuras para las que esconder información.

## Parámetros

| Parámetro | `conservador` | `equilibrado` | `agresivo` |
|---|---|---|---|
| Umbral truco / retruco / vale cuatro | 0,70 / 0,80 / 0,90 | 0,65 / 0,75 / 0,85 | 0,55 / 0,65 / 0,75 |
| `farol` | 0 | 0,05 | 0,15 |
| `lectura` | 1 | 1 | 0,5 |
| `margen` para aceptar | +0,10 | 0 | −0,05 |

## Enfrentamientos

Puntos por mano del rival de la fila contra el de la columna, ± el error estándar, con el mazo
completo. Se jugaron 1000 repartos duplicados por par (`truco/evaluation/arena.py`): cada reparto
se juega dos veces con los asientos invertidos, así la suerte de las cartas se cancela.

| | vs `aleatorio` | vs `conservador` | vs `equilibrado` | vs `agresivo` |
|---|---|---|---|---|
| `conservador` | +0,89 ± 0,03 | — | +0,12 ± 0,015 | +0,29 ± 0,025 |
| `equilibrado` | +1,09 ± 0,03 | −0,12 ± 0,015 | — | +0,25 ± 0,02 |
| `agresivo` | +1,46 ± 0,03 | −0,29 ± 0,025 | −0,25 ± 0,02 | — |

Todas las diferencias superan con holgura dos errores estándar: el orden conservador > equilibrado >
agresivo no es suerte de las cartas.

## Limitaciones

- La equity supone que, después de la decisión actual, nadie vuelve a cantar. Ignora el valor de
  poder subir más tarde.
- El juego de cartas no depende de los cantos: no tira una carta mala para disimular.
- La lectura trata igual un canto y un "quiero", y no saca nada de que el rival *no* cante.
