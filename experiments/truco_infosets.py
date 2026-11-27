"""Counts the information sets of one hand of truco, to size a tabular MCCFR.

Tabular CFR and MCCFR store, for every information set, one regret and one average-strategy entry
per legal action. Sampling makes the time per iteration independent of the size of the tree, but
the memory still grows with the number of information sets, so this number decides whether the
full deck can be solved without a neural network.

How it counts without walking the whole game tree: an information set of player p is (who is
mano, p's hand, public history). For each mano and each hand of p, the script walks the tree of
histories that p can see, where the opponent may play any card level that is still possible given
p's hand and the cards the opponent already played. Each node of that tree where p acts is one
information set. Subtrees that start from the same state are counted once (memoization), which
makes the count fast even when the result is large.

`--check` validates the count against a brute-force walk of the OpenSpiel game that collects the
distinct information state strings (only feasible with few levels, e.g. `--niveles 5`).

Logs to MLflow (experiment "truco-infosets") the number of information sets, of (information set,
action) pairs, and the memory a tabular MCCFR would need.

Usage, from the repo root:
    python experiments/truco_infosets.py --niveles 14
    python experiments/truco_infosets.py --niveles 5 --check
"""

import argparse
import time

import mlflow
import pyspiel

import truco.games.truco  # noqa: F401  (registers the game)
from truco.games.cards import NUM_TRUCO_LEVELS, TRUCO_LEVEL_CARDS
from truco.games.truco import TrucoState, _AnyCards, hands
from truco.tracking import setup_experiment

# Two tables per information set and action (cumulative regret and average strategy), float32.
BYTES_PER_ACTION = 2 * 4


def state_key(state: TrucoState, opponent_played: tuple[int, ...]) -> tuple:
    """Everything that determines the subtree below `state`, from the viewpoint of one player."""
    return (
        tuple(sorted(state.remaining[0])),
        tuple(sorted(state.remaining[1])),
        opponent_played,
        tuple(state.results),
        tuple(state.table),
        state.points,
        state.quiero,
        state.pending,
        state.caller,
        tuple(state.window),
        state.card_player,
        state.is_terminal(),
    )


def count_for_hand(game, mano: int, player: int, hand: tuple[int, ...]) -> tuple[int, int]:
    """(information sets, information set-action pairs) of `player` holding `hand`."""
    opponent = 1 - player
    root = TrucoState(game)
    root.mano = mano
    root.hands = [(), ()]
    root.hands[player] = hand
    root.remaining = [_AnyCards(), _AnyCards()]
    root.remaining[player] = list(hand)
    root.window = [mano, 1 - mano]
    root.card_player = mano

    def opponent_cards(opponent_played: tuple[int, ...]) -> list[int]:
        """Card levels the opponent may still hold, given what is known to `player`."""
        return [
            level
            for level in range(game.niveles)
            if hand.count(level) + opponent_played.count(level) < len(TRUCO_LEVEL_CARDS[level])
        ]

    memo: dict[tuple, tuple[int, int]] = {}

    def walk(state: TrucoState, opponent_played: tuple[int, ...]) -> tuple[int, int]:
        if state.is_terminal():
            return 0, 0
        key = state_key(state, opponent_played)
        if key in memo:
            return memo[key]
        actor = state.current_player()
        plays_card = state.pending is None and not state.window
        if actor == opponent and plays_card:
            actions = opponent_cards(opponent_played)
        else:
            actions = state.legal_actions()
        infosets, pairs = (1, len(actions)) if actor == player else (0, 0)
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            played = opponent_played
            if actor == opponent and plays_card:
                played = tuple(sorted((*opponent_played, action)))
            sub_infosets, sub_pairs = walk(child, played)
            infosets += sub_infosets
            pairs += sub_pairs
        memo[key] = (infosets, pairs)
        return memo[key]

    return walk(root, ())


def count_infosets(game) -> tuple[int, int]:
    infosets = pairs = 0
    for mano in range(2):
        for player in range(2):
            for hand in hands(game.niveles):
                i, p = count_for_hand(game, mano, player, hand)
                infosets += i
                pairs += p
    return infosets, pairs


def brute_force(game) -> tuple[int, int]:
    """Same count, walking every node of the OpenSpiel game tree."""
    seen: dict[str, int] = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        seen[state.information_state_string()] = len(state.legal_actions())
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return len(seen), sum(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--niveles", type=int, default=NUM_TRUCO_LEVELS)
    parser.add_argument("--check", action="store_true", help="compare with a brute-force walk")
    args = parser.parse_args()

    game = pyspiel.load_game("truco", {"niveles": args.niveles})
    setup_experiment("truco-infosets")
    with mlflow.start_run(run_name=f"niveles={args.niveles}"):
        mlflow.log_params({"niveles": args.niveles, "check": args.check})

        start = time.perf_counter()
        infosets, pairs = count_infosets(game)
        seconds = time.perf_counter() - start
        memory_gb = pairs * BYTES_PER_ACTION / 1e9
        print(f"niveles={args.niveles}: {infosets:,} information sets, {pairs:,} pairs")
        print(f"tabular MCCFR memory (2 float32 tables): {memory_gb:.2f} GB  [{seconds:.0f} s]")
        mlflow.log_metrics(
            {
                "infosets": infosets,
                "infoset_actions": pairs,
                "memory_gb": memory_gb,
                "seconds": seconds,
            }
        )

        if args.check:
            start = time.perf_counter()
            expected = brute_force(game)
            print(
                f"brute force: {expected[0]:,} information sets, {expected[1]:,} pairs "
                f"[{time.perf_counter() - start:.0f} s]"
            )
            mlflow.log_metric("check_ok", int(expected == (infosets, pairs)))
            if expected != (infosets, pairs):
                raise SystemExit("count does not match the brute-force walk")
            print("check ok")


if __name__ == "__main__":
    main()
