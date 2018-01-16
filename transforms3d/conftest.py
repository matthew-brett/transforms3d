""" Configuration for py.test test run
"""

def pytest_ignore_collect(path, config):
    """ Skip the origin Gohlke transforms for doctests.

    That file needs some specific doctest setup.
    """
    return path.basename == '_gohlketransforms.py'
