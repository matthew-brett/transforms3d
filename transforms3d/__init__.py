''' transforms3d package

Based on, largely using transformations.py by Christoph Gohlke

Additional code monkey work by Matthew Brett
'''

from . import taitbryan
from . import affines
from . import quaternions
from . import euler
from . import axangles
from . import reflections
from . import shears
from . import zooms

from . import _version
__version__ = _version.get_versions()['version']
