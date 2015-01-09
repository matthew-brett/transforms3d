''' transforms3d package

Based on, largely using transformations.py by Christoph Gohlke

Additional code monkey work by Matthew Brett
'''

from . import taitbryan
from . import affines
from . import quaternions

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
