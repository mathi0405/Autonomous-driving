"""Exercise malformed evidence and interrupted evaluation lifetimes."""

from dataclasses import asdict

import pytest

from ad_rl.validation import runner
from ad_rl.validation.policies import policy_identity
from ad_rl.validation.scenarios import Scenario


def test_environment_closes_when_policy_loading_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(runner.ScenarioEnvironment, "close", lambda self: closed.append(True))

    def fail(*args):
        raise ValueError("model could not be loaded")

    monkeypatch.setattr(runner, "make_policy", fail)
    with pytest.raises(ValueError, match="model could"):
        runner.run_episode(Scenario("case", 1), {"name": "model"})
    assert closed == [True]


def test_changed_source_cannot_publish_complete_run(tmp_path, monkeypatch):
    case = Scenario("case", 1, max_steps=2)
    suite = {"schema": 1, "scenarios": [asdict(case)]}
    fingerprints = iter(["start", "changed"])
    monkeypatch.setattr(runner, "source_fingerprint", lambda: next(fingerprints))
    with pytest.raises(ValueError, match="Source changed"):
        runner.run_suite(suite, [case], policy_identity("stanley"), tmp_path / "run")
    assert not (tmp_path / "run/run.json").exists()


@pytest.mark.parametrize(
    "alter",
    [
        lambda m: m.update(success=True),
        lambda m: m.update(collision=True),
        lambda m: m.update(timeout=1),
        lambda m: m.update(terminal_reason="RUNNING"),
        lambda m: m.update(route_completion=float("nan")),
        lambda m: m.update(route_completion=1.2),
    ],
)
def test_inconsistent_outcomes_cannot_be_aggregated(alter):
    episode = runner.run_episode(Scenario("case", 1, max_steps=2), policy_identity("stanley"))
    alter(episode["metrics"])
    with pytest.raises(ValueError, match="outcome|metric"):
        runner.summarize([episode])


def test_suite_mapping_is_checked_before_execution(tmp_path):
    case = Scenario("case", 1)
    suite = {"schema": 1, "scenarios": [asdict(Scenario("other", 2))]}
    with pytest.raises(ValueError, match="Suite mapping"):
        runner.run_suite(suite, [case], policy_identity("stanley"), tmp_path / "run")
    assert not (tmp_path / "run").exists()
