import numpy as np
import pyspiel
import pytest

import truco.games.truco  # noqa: F401  (registra el juego)
from truco.agents.truco_rivales import CONSERVADOR, RIVALES, HeuristicPolicy
from truco.evaluation.arena import crossplay, play_match, sample_actions

# Dos heurísticas sin faroles: juegan de forma determinística.
OTRO_DETERMINISTA = HeuristicPolicy("otro", (0.6, 0.7, 0.8), farol=0.0)


@pytest.fixture(scope="module")
def game():
    return pyspiel.load_game("truco", {"niveles": 14})


def test_sample_actions_follows_probabilities():
    rng = np.random.default_rng(0)
    probs = np.tile([0.0, 0.25, 0.0, 0.75], (20000, 1))
    counts = np.bincount(sample_actions(probs, rng), minlength=4)
    assert counts[0] == counts[2] == 0
    assert counts[3] / counts.sum() == pytest.approx(0.75, abs=0.02)


def test_deterministic_policy_against_itself_is_exactly_zero(game):
    result = play_match(game, CONSERVADOR, CONSERVADOR, num_deals=100)
    assert result.mean == 0.0
    assert result.stderr == 0.0


def test_swapping_policies_negates_result(game):
    ab = play_match(game, CONSERVADOR, OTRO_DETERMINISTA, num_deals=100, seed=3)
    ba = play_match(game, OTRO_DETERMINISTA, CONSERVADOR, num_deals=100, seed=3)
    assert ab.mean == pytest.approx(-ba.mean)


def test_heuristic_beats_random(game):
    result = play_match(game, RIVALES["equilibrado"], RIVALES["aleatorio"], num_deals=200)
    assert result.mean > 3 * result.stderr > 0


def test_crossplay_is_antisymmetric(game):
    policies = [RIVALES["aleatorio"], CONSERVADOR, OTRO_DETERMINISTA]
    means, stderrs = crossplay(game, policies, num_deals=50)
    assert means.shape == stderrs.shape == (3, 3)
    assert np.allclose(means, -means.T)
    assert np.allclose(np.diag(means), 0)
