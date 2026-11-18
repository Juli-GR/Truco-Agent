import numpy as np
import pytest

import truco.games.truco  # noqa: F401  (registers the game)
from truco.games.truco import sample_deal


@pytest.fixture
def make_states():
    """Factory: `make_states(game, n)` gives up to `n` states at a decision of some player, after a
    deal and one or two random actions."""

    def make(game, n, seed=0):
        rng = np.random.default_rng(seed)
        states = []
        for _ in range(n):
            state = game.new_initial_state()
            for action in sample_deal(game, rng):
                state.apply_action(action)
            for _ in range(rng.integers(1, 3)):
                state.apply_action(int(rng.choice(state.legal_actions())))
                if state.is_terminal():
                    break
            if not state.is_terminal():
                states.append(state)
        return states

    return make
