from gym_super_mario_bros import actions


def _is_action(action):
    return isinstance(action, list) and all(isinstance(btn, str) for btn in action)


def test_action_sets_are_lists_of_button_lists():
    for name in ["RIGHT_ONLY", "SIMPLE_MOVEMENT", "COMPLEX_MOVEMENT"]:
        action_set = getattr(actions, name)
        assert isinstance(action_set, list)
        assert len(action_set) > 0
        assert all(_is_action(a) for a in action_set)


def test_action_sets_start_with_noop():
    assert actions.RIGHT_ONLY[0] == ["NOOP"]
    assert actions.SIMPLE_MOVEMENT[0] == ["NOOP"]
    assert actions.COMPLEX_MOVEMENT[0] == ["NOOP"]


def test_complex_movement_is_supersetish():
    # Every action in RIGHT_ONLY should appear in SIMPLE and COMPLEX.
    for a in actions.RIGHT_ONLY:
        assert a in actions.SIMPLE_MOVEMENT
        assert a in actions.COMPLEX_MOVEMENT
