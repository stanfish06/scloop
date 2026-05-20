from anndata import AnnData

from scloop.benchmarking.datasets import RealData, RealDataMeta


def test_list_available_includes_dummy():
    assert "dummy_test" in RealData.list_available()


def test_hydrate_from_hub():
    d = RealData(data=None, meta=RealDataMeta(), name="dummy_test")
    d.hydrate()
    assert isinstance(d.data, AnnData)
    assert d.meta.dataset_name == "dummy_test"
    assert d.meta.doi is None
    assert d.meta.paper_title is None


def test_hydrate_local_override(tmp_path):
    import yaml
    from anndata import AnnData as A

    A().write_h5ad(tmp_path / "data.h5ad")
    (tmp_path / "meta.yaml").write_text(
        yaml.safe_dump(
            {"dataset_name": "dummy_test", "doi": "10.x/y", "paper_title": "T"}
        )
    )
    d = RealData(
        data=None,
        meta=RealDataMeta(),
        name="dummy_test",
        local_override=str(tmp_path),
    )
    d.hydrate()
    assert isinstance(d.data, AnnData)
    assert d.meta.doi == "10.x/y"
    assert d.meta.paper_title == "T"
