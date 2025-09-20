def test_imports():
    import terra_ai
    from terra_ai.cli import app
    assert app is not None
