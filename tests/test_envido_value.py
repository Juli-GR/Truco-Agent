import pyspiel
import pytest

from truco.evaluation.envido import expected_value_by_tantos
from truco.games.envido import Action


@pytest.fixture
def game():
    import truco.games.envido  # noqa: F401  (registra el juego)

    return pyspiel.load_game("truco_envido")


def test_nobody_calls_is_worth_zero(game):
    by_tantos, total = expected_value_by_tantos(game, lambda s: {Action.PASO: 1.0})
    assert total == {"mano": 0.0, "pie": 0.0}
    assert all(v == 0.0 for role in by_tantos.values() for v in role.values())


def test_always_envido_and_quiero(game):
    def strategy(state):
        return {Action.QUIERO: 1.0} if state.calls else {Action.ENVIDO: 1.0}

    by_tantos, total = expected_value_by_tantos(game, strategy)
    assert by_tantos["mano"][33] == pytest.approx(2.0)  # 33 siendo mano nunca pierde
    assert by_tantos["pie"][0] == pytest.approx(-2.0)  # 0 siendo pie nunca gana
    assert total["mano"] > 0  # el mano gana los empates
    assert total["mano"] == pytest.approx(-total["pie"])
