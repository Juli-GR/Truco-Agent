import pyspiel
from open_spiel.python.algorithms import cfr, exploitability


def test_cfr_converges_on_kuhn_poker():
    """Verifica que OpenSpiel funciona: CFR sobre Kuhn poker debe acercarse al equilibrio."""
    game = pyspiel.load_game("kuhn_poker")
    solver = cfr.CFRSolver(game)
    for _ in range(300):
        solver.evaluate_and_update_policy()
    assert exploitability.exploitability(game, solver.average_policy()) < 0.01
