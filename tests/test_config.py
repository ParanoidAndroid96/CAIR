from cair.config import resolve_experiment


TASKS = [
    "fetch_pick_and_place", "fetch_slide", "fetch_push",
    "adroit_hammer", "adroit_pen",
    "halfcheetah", "hopper", "walker2d", "ant", "swimmer", "humanoid",
]


def test_all_main_pairings_resolve_for_all_tasks():
    for task in TASKS:
        for il in ("bc", "dagger"):
            for rl in ("sac", "ddpg", "ppo"):
                exp = resolve_experiment(task, il, rl)
                assert exp["pairing"] == f"{il}+{rl}"
                assert exp["train_steps"] > 0
                assert "learning_rate" in exp["rl_config"]


def test_benchmark_specific_cair_sync_interval():
    assert resolve_experiment("fetch_push", "bc", "sac")["cair"]["sync_interval"] == 10
    assert resolve_experiment("adroit_hammer", "dagger", "ppo")["cair"]["sync_interval"] == 10
    assert resolve_experiment("ant", "bc", "sac")["cair"]["sync_interval"] == 5


def test_paper_specific_settings():
    push = resolve_experiment("fetch_push", "bc", "sac")
    assert push["rl_config"]["learning_rate"] == 3.0e-5
    assert push["rl_config"]["batch_size"] == 256

    swimmer = resolve_experiment("swimmer", "bc", "ddpg")
    assert swimmer["rl_config"]["gamma"] == 0.9999

    humanoid = resolve_experiment("humanoid", "dagger", "ppo")
    assert humanoid["rl_config"]["clip_range"] == 0.3
