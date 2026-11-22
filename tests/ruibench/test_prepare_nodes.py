"""prepare's ``node:`` part selector for Objaverse sources (v3): a model's
named glTF nodes, kept before converting -- the converter merges a GLB into
one mesh, so a node cannot be picked by prim path afterwards."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ruibench.core.errors import ConfigError  # noqa: E402
from ruibench.engines.isaac import prepare  # noqa: E402


def _two_boxes(path):
    """A GLB with two nodes: "base" at the origin, "jar" 2 units up."""
    scene = trimesh.Scene()
    base = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    jar = trimesh.creation.box(extents=(1.0, 1.0, 3.0))
    up = trimesh.transformations.translation_matrix((0.0, 2.0, 0.0))
    scene.add_geometry(base, node_name="base", geom_name="base_mesh")
    scene.add_geometry(jar, node_name="jar", geom_name="jar_mesh", transform=up)
    scene.export(path)
    return path


def test_the_named_nodes_are_kept_where_they_were(tmp_path):
    glb = _two_boxes(tmp_path / "model.glb")
    out = prepare._keep_nodes(glb, ["jar"])
    assert out.exists() and out.parent == glb.parent and out != glb
    kept = trimesh.load(out, force="scene")
    assert len(kept.geometry) == 1
    # still 2 units up: the part keeps its place in the model's own units
    lo, hi = kept.bounds
    assert np.allclose(lo, (-0.5, 1.5, -1.5)) and np.allclose(hi, (0.5, 2.5, 1.5))
    # asked again: the file made the first time
    assert prepare._keep_nodes(glb, ["jar"]) == out


def test_a_baked_node_carries_its_transform_in_its_vertices(tmp_path):
    """v10: a kept node's transform went through the converter twice -- a
    rattan basket scaled 3.42 across and 1.49 up came out 3.7 cm deep, not
    8.4.  Baked, the node is at rest and its vertices are where it drew them."""
    scene = trimesh.Scene()
    pot = trimesh.creation.cylinder(radius=1.0, height=2.0)
    squash = np.diag([3.0, 1.5, 3.0, 1.0])
    scene.add_geometry(pot, node_name="pot", geom_name="pot_mesh", transform=squash)
    glb = tmp_path / "pot.glb"
    scene.export(glb)
    out = prepare._keep_nodes(glb, ["pot"], bake=True)
    assert out.name.endswith(".baked.glb") and out != prepare._keep_nodes(glb, ["pot"])
    kept = trimesh.load(out, force="scene")
    (node,) = kept.graph.nodes_geometry
    transform, geometry = kept.graph[node]
    assert np.allclose(transform, np.eye(4))
    assert np.allclose(kept.geometry[geometry].extents, trimesh.load(glb, force="scene").extents)


def test_several_nodes_and_a_missing_one(tmp_path):
    glb = _two_boxes(tmp_path / "model.glb")
    both = trimesh.load(prepare._keep_nodes(glb, ["base", "jar"]), force="scene")
    assert len(both.geometry) == 2
    with pytest.raises(ConfigError, match="no geometry node"):
        prepare._keep_nodes(glb, ["lid"])
    with pytest.raises(ConfigError):
        prepare._keep_nodes(glb, [])


def test_an_objaverse_entry_converts_only_its_nodes(tmp_path, monkeypatch):
    glb = _two_boxes(tmp_path / "abc.glb")
    monkeypatch.setattr(prepare, "_download", lambda url, dest: glb)
    converted = []
    monkeypatch.setattr(prepare, "_convert", lambda src, out: converted.append(src) or out)

    class Entry:
        name, source, part = "toy_part", "objaverse:000-001/abc", "node:jar"

    usd, prim = prepare.source_usd(Entry)
    assert prim == ""                                  # the sub-GLB is the whole source
    assert converted and converted[0].name.startswith("abc.nodes-jar")
    Entry.part = ""
    usd, prim = prepare.source_usd(Entry)
    assert converted[-1] == glb and prim == ""
