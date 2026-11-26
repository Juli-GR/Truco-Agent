"""Exportación de juegos de OpenSpiel al formato EFG de Gambit.

OpenSpiel carga un archivo EFG como el juego `efg_game`, implementado en C++. Así los algoritmos de
C++ recorren el árbol sin llamar al juego en Python en cada nodo, que es el cuello de botella con
los juegos escritos en Python. Es el mismo formato que escribe efg_writer.cc de OpenSpiel, que no
está expuesto en Python.

Para que el archivo y la memoria de C++ sean chicos, las acciones se nombran con su número en el
juego original y los information sets se identifican solo con un número. `EFGMapping` guarda la
correspondencia con el juego original.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pyspiel


@dataclass
class EFGMapping:
    """Correspondencia entre el juego EFG y el original."""

    # Por jugador, el information state string original del information set número i + 1.
    infosets: list[list[str]]
    # Acción original de cada id de acción de jugador en efg_game.
    actions: list[int]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EFGMapping":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def export_efg(game: pyspiel.Game, path: Path) -> EFGMapping:
    """Escribe el árbol completo de `game` en `path`."""
    infosets = [{} for _ in range(game.num_players())]
    # efg_game numera las acciones de jugador en el orden en que aparecen por primera vez.
    action_ids = {}
    counters = {"chance": 0, "terminal": 0}
    players = " ".join(f'"Jugador {p}"' for p in range(game.num_players()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'EFG 2 R "{game}" {{ {players} }}\n')
        _write_subtree(f, game.new_initial_state(), infosets, action_ids, counters)
    return EFGMapping(
        infosets=[list(by_string) for by_string in infosets], actions=list(action_ids)
    )


def _write_subtree(f, state, infosets, action_ids, counters) -> None:
    if state.is_terminal():
        counters["terminal"] += 1
        returns = " ".join(repr(r) for r in state.returns())
        f.write(f't "" {counters["terminal"]} "" {{ {returns} }}\n')
        return

    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        counters["chance"] += 1
        body = " ".join(f'"{a}" {p!r}' for a, p in outcomes)
        f.write(f'c "" {counters["chance"]} {{ {body} }} 0\n')
        actions = [a for a, _ in outcomes]
    else:
        player = state.current_player()
        by_string = infosets[player]
        number = by_string.setdefault(state.information_state_string(), len(by_string) + 1)
        # efg_game gives each action name a global id in order of first appearance, but in each
        # node it pairs the node's ids, sorted, with the node's actions in file order. Both orders
        # only agree if every node lists its actions by increasing id: the ones already seen first,
        # by id, then the new ones (which get the next ids in that same order).
        actions = sorted(state.legal_actions(), key=lambda a: action_ids.get(a, len(action_ids)))
        for a in actions:
            action_ids.setdefault(a, len(action_ids))
        names = " ".join(f'"{a}"' for a in actions)
        f.write(f'p "" {player + 1} {number} {{ {names} }} 0\n')

    for action in actions:
        _write_subtree(f, state.child(action), infosets, action_ids, counters)


def load_efg(path: Path) -> pyspiel.Game:
    return pyspiel.load_game("efg_game", {"filename": str(path)})


def infoset_number(efg_state: pyspiel.State) -> int:
    """Número del information set del jugador que juega en `efg_state`."""
    # El string de efg_game es "jugador-observador-número-nombre".
    return int(efg_state.information_state_string().split("-")[2])


def original_strategy(
    policy: pyspiel.Policy, mapping: EFGMapping
) -> Iterator[tuple[int, str, dict[int, float]]]:
    """Estrategia de una política del juego EFG en términos del juego original.

    Genera (jugador, information state string original, probabilidad de cada acción original).
    """
    for player, strings in enumerate(mapping.infosets):
        for number, info_state in enumerate(strings, start=1):
            probs = policy.get_state_policy(f"{player}-{player}-{number}-")
            yield player, info_state, {mapping.actions[a]: p for a, p in probs}
