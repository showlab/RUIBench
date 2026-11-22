"""Where a result comes from, and which results do not compare with the rest."""

from ruikit.experiments import provenance


def test_the_commit_is_read_from_git_itself(tmp_path):
    git = tmp_path / "repo" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert provenance.commit(tmp_path / "repo") == "a" * 40
    # a packed ref; a worktree's .git file; a detached head; no git at all
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack-refs with: peeled\n" + "b" * 40 + " refs/heads/main\n")
    assert provenance.commit(tmp_path / "repo") == "b" * 40
    (git / "worktrees" / "wt").mkdir(parents=True)
    (git / "worktrees" / "wt" / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "worktrees" / "wt" / "commondir").write_text("../..\n")
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text(f"gitdir: {git / 'worktrees' / 'wt'}\n")
    assert provenance.commit(tmp_path / "wt") == "b" * 40
    (git / "HEAD").write_text("c" * 40 + "\n")
    assert provenance.commit(tmp_path / "repo") == "c" * 40
    assert provenance.commit(tmp_path / "nowhere") == "unknown"


def test_a_result_compares_on_the_frozen_version_in_a_fresh_simulator():
    fresh = {**provenance.stamp(1, 1), "frozen": "v2.0", "freeze_clean": True}
    assert set(fresh) == set(provenance.FIELDS) and fresh["python"] and fresh["numpy"]
    assert provenance.not_comparable({"provenance": fresh}, "v2.0") == []
    later = {**fresh, "process_episode": 3, "per_worker": 6}
    assert provenance.not_comparable({"provenance": later}, "v2.0") == [
        "episode 3 of its simulator process"]
    assert provenance.not_comparable({"provenance": {**fresh, "frozen": "v1.17"}}, "v2.0") == [
        "frozen version v1.17, not v2.0"]
    assert provenance.not_comparable({"provenance": {**fresh, "freeze_clean": False}}, "v2.0") == [
        "the checkout differed from v2.0"]
    # what is not recorded counts as unknown, and is flagged
    assert provenance.not_comparable({}, "v2.0") == ["no provenance"]
    partial = {k: v for k, v in fresh.items() if k not in ("process_episode", "numpy")}
    assert provenance.not_comparable({"provenance": partial}, "v2.0") == [
        "process_episode, numpy unknown"]
    # a commit that could not be read is recorded as such, not flagged
    assert provenance.not_comparable({"provenance": {**fresh, "commit": "unknown"}}, "v2.0") == []


def test_a_runner_says_what_it_runs():
    line = provenance.describe(1)
    assert line.startswith("code: RUIKit ") and "numpy" in line and "not comparable" not in line
    assert "up to 6 episodes per simulator" in provenance.describe(6)


def test_the_report_lists_the_results_that_do_not_compare():
    from ruikit.report import tables

    fresh = {**provenance.stamp(1, 1), "frozen": provenance.expected_version(),
             "freeze_clean": True}
    row = dict(agent="loop-m", harness="loop", model="m", task="WK1", suite="knowledge",
               level="easy", seed=0, profile="rui.pose/v1", column="pose", aids="none",
               success=True, replies=3, sim_seconds=10.0, version="v2.0")
    rows = [dict(row, folder="o/loop-m/WK1_easy_s0_rui.pose-v1", provenance=fresh),
            dict(row, seed=1, folder="o/loop-m/WK1_easy_s1_rui.pose-v1",
                 provenance=dict(fresh, process_episode=2, per_worker=2)),
            dict(row, seed=2, folder="o/loop-m/WK1_easy_s2_rui.pose-v1")]
    assert [why for _, why in tables.not_comparable(rows)] == [
        ["episode 2 of its simulator process"], ["no provenance"]]
    text = tables.report("All results", rows)
    assert "**2 of 3 results are not comparable**" in text
    assert "| loop-m | WK1_easy_s1_rui.pose-v1 | episode 2 of its simulator process |" in text
    assert "| loop-m | WK1_easy_s2_rui.pose-v1 | no provenance |" in text
    assert "WK1_easy_s0" not in text.split("## Not comparable")[1]


def test_a_version_that_changed_one_task_keeps_the_others_comparable():
    """v2.3.1 changed CL3, WK1 and DI4 and nothing else: a v2.3 result of any
    other task still compares with it; one of those three does not."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3", "freeze_clean": True}
    assert provenance.not_comparable({"task": "CR1", "provenance": old}, "v2.3.1") == []
    for task in ("CL3", "WK1", "DI4"):
        assert provenance.not_comparable({"task": task, "provenance": old}, "v2.3.1") == [
            "frozen version v2.3, not v2.3.1"]
    # only the way forward, and only between those two
    assert provenance.not_comparable({"task": "CR1", "provenance": {**old, "frozen": "v2.3.1"}},
                                     "v2.3") == ["frozen version v2.3.1, not v2.3"]
    assert provenance.not_comparable({"task": "CR1", "provenance": {**old, "frozen": "v2.2"}},
                                     "v2.3.1") == ["frozen version v2.2, not v2.3.1"]


def test_changes_add_up_along_a_chain_of_versions():
    """v2.3.2 changed no task: v2.3.1's results all compare with it, and v2.3's
    but those of the three tasks v2.3.1 changed."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3", "freeze_clean": True}
    v231 = {**old, "frozen": "v2.3.1"}
    assert provenance.changed_between("v2.3", "v2.3.2") == {"CL3", "WK1", "DI4"}
    assert provenance.changed_between("v2.3.1", "v2.3.2") == frozenset()
    assert provenance.changed_between("v2.2", "v2.3.2") is None
    assert provenance.not_comparable({"task": "CL3", "provenance": v231}, "v2.3.2") == []
    assert provenance.not_comparable({"task": "CR1", "provenance": old}, "v2.3.2") == []
    assert provenance.not_comparable({"task": "DI4", "provenance": old}, "v2.3.2") == [
        "frozen version v2.3, not v2.3.2"]


def test_v2_3_3_changed_cr2_only():
    """v2.3.3 judged CR2's bulb by its cap: v2.3.2's results compare with it on
    every other task, and v2.3's on all but four."""
    v232 = {**provenance.stamp(1, 1), "frozen": "v2.3.2", "freeze_clean": True}
    assert provenance.changed_between("v2.3.2", "v2.3.3") == {"CR2"}
    assert provenance.changed_between("v2.3", "v2.3.3") == {"CL3", "WK1", "DI4", "CR2"}
    assert provenance.not_comparable({"task": "CR1", "provenance": v232}, "v2.3.3") == []
    assert provenance.not_comparable({"task": "CR2", "provenance": v232}, "v2.3.3") == [
        "frozen version v2.3.2, not v2.3.3"]
