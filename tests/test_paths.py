from baed import paths


def test_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BAED_ROOT", str(tmp_path))
    assert paths.graph_file("x.pkl") == tmp_path / "data" / "graphs" / "x.pkl"
    assert paths.run_dir("elliptic", "r1") == tmp_path / "outputs/runs/elliptic/multinomial_diffusion/multistep/r1"
    p = paths.ensure(paths.gae_weight("a.pt"))
    assert p.parent.is_dir()
