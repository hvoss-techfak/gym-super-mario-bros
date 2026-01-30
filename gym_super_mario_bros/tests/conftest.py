import os
import pytest


def pytest_collection_modifyitems(config, items):
    # If RUN_INTEGRATION isn't set, ensure integration tests are skipped even
    # when selected accidentally.
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip_integration = pytest.mark.skip(reason="Set RUN_INTEGRATION=1 to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
