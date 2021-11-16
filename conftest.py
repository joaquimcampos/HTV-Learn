def pytest_addoption(parser):
    """Pytest parser"""
    parser.addoption("--plot",
                     action="store",
                     help="list of pytest fixtures to plot")
    parser.addoption('--seed',
                     action='store',
                     default=42,
                     help="Set seed")
