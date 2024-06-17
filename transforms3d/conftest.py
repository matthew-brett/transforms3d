""" Configuration for py.test test run
"""

def pytest_ignore_collect(collection_path, config):
    """ Skip the origin Gohlke transforms for doctests.

    That file needs some specific doctest setup.
    """
    return collection_path.name == '_gohlketransforms.py'
