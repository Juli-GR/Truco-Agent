import pyspiel
import pytest
from open_spiel.python import policy as policy_lib

from truco.games.efg import EFGMapping, export_efg, infoset_number, load_efg, original_strategy


@pytest.fixture
def envido_and_efg(tmp_path):
    import truco.games.envido  # noqa: F401  (registra el juego)

    game = pyspiel.load_game("truco_envido")
    path = tmp_path / "envido.efg"
    mapping = export_efg(game, path)
    return game, load_efg(path), mapping


def test_same_infosets(envido_and_efg):
    game, efg, mapping = envido_and_efg
    num_infosets = sum(len(p) for p in mapping.infosets)
    assert num_infosets == len(policy_lib.TabularPolicy(game).state_lookup)
    assert num_infosets == len(policy_lib.TabularPolicy(efg).state_lookup)


def test_same_exploitability_of_uniform_policy(envido_and_efg):
    game, efg, _ = envido_and_efg
    expl = pyspiel.exploitability(game, pyspiel.UniformRandomPolicy(game))
    assert pyspiel.exploitability(efg, pyspiel.UniformRandomPolicy(efg)) == pytest.approx(expl)


def test_mapping_back_to_original(envido_and_efg):
    game, efg, mapping = envido_and_efg
    state, efg_state = game.new_initial_state(), efg.new_initial_state()
    while state.is_chance_node():
        action = state.chance_outcomes()[-1][0]
        efg_action = next(
            a for a in efg_state.legal_actions() if int(efg_state.action_to_string(a)) == action
        )
        state.apply_action(action)
        efg_state.apply_action(efg_action)
    player = state.current_player()
    assert mapping.infosets[player][infoset_number(efg_state) - 1] == (
        state.information_state_string()
    )
    efg_actions = efg_state.legal_actions()
    assert all(mapping.actions[a] == int(efg_state.action_to_string(a)) for a in efg_actions)


def test_mapping_holds_in_every_node(tmp_path):
    # In Leduc poker, as in truco, some nodes have actions that the first node does not have
    # (fold), which is what used to break the action mapping.
    game = pyspiel.load_game("leduc_poker")
    mapping = export_efg(game, tmp_path / "leduc.efg")
    efg = load_efg(tmp_path / "leduc.efg")

    def walk(state, efg_state):
        if state.is_terminal():
            assert efg_state.returns() == pytest.approx(state.returns())
            return
        if state.is_chance_node():
            for efg_action in efg_state.legal_actions():
                action = int(efg_state.action_to_string(efg_action))
                walk(state.child(action), efg_state.child(efg_action))
            return
        player = state.current_player()
        assert mapping.infosets[player][infoset_number(efg_state) - 1] == (
            state.information_state_string()
        )
        assert sorted(mapping.actions[a] for a in efg_state.legal_actions()) == (
            state.legal_actions()
        )
        for efg_action in efg_state.legal_actions():
            walk(state.child(mapping.actions[efg_action]), efg_state.child(efg_action))

    walk(game.new_initial_state(), efg.new_initial_state())


def test_original_strategy(envido_and_efg, tmp_path):
    game, efg, mapping = envido_and_efg
    mapping.save(tmp_path / "mapping.json")
    mapping = EFGMapping.load(tmp_path / "mapping.json")
    rows = list(original_strategy(pyspiel.UniformRandomPolicy(efg), mapping))
    assert len(rows) == len(policy_lib.TabularPolicy(game).state_lookup)
    state = game.new_initial_state()
    for a in [0, 30 * 34 + 25]:  # mano 0 y tantos (30, 25)
        state.apply_action(a)
    probs = next(p for _, s, p in rows if s == state.information_state_string())
    assert probs == pytest.approx({a: 1 / 3 for a in state.legal_actions()})
