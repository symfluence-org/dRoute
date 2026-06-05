"""Test basic package imports and metadata."""

import pytest


def test_droute_import():
    """Test that droute can be imported."""
    import droute
    assert droute is not None


def test_version():
    """Test that version is accessible and matches the single source of truth."""
    import re

    import droute
    from droute import _version

    assert hasattr(droute, '__version__')
    assert isinstance(droute.__version__, str)
    # Exposed version comes from droute._version (the source of truth pyproject also reads).
    assert droute.__version__ == _version.__version__
    # ...and is a well-formed version string (avoids a hardcoded literal going stale on bumps).
    assert re.match(r'^\d+\.\d+\.\d+', droute.__version__), droute.__version__


def test_author():
    """Test that author is accessible."""
    import droute
    assert hasattr(droute, '__author__')
    assert droute.__author__ == "Darri Eythorsson"


@pytest.mark.skip(reason="Backwards compatibility test conflicts with pybind11 type registration in same session")
def test_backwards_compatibility():
    """Test that pydmc_route shim works with deprecation warning.

    Note: This test is skipped because pybind11 doesn't allow the same types
    to be registered twice in a single Python session. The backwards compatibility
    shim works correctly when used in isolation.
    """
    import warnings
    import sys

    # Clean up any previous imports of pydmc_route
    if 'pydmc_route' in sys.modules:
        del sys.modules['pydmc_route']

    # Capture warnings and import
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pydmc_route

        # Check that a deprecation warning was issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, f"Expected DeprecationWarning, got {[str(x.message) for x in w]}"

        # Verify the module works
        assert pydmc_route is not None
        assert hasattr(pydmc_route, '__version__')
        assert hasattr(pydmc_route, 'Network')


def test_core_classes_available():
    """Test that core classes are available."""
    import droute

    # These should be available from the C++ extension
    expected_classes = ['Network', 'Reach', 'RouterConfig']

    for cls_name in expected_classes:
        assert hasattr(droute, cls_name), f"Missing class: {cls_name}"
