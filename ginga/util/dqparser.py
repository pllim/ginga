"""Module to handle parsing of data quality (DQ) flags."""

# STDLIB
import os

# THIRD-PARTY
import numpy as np
from astropy.io import ascii

__all__ = ['DQParser']


# STScI reftools.interpretdq.DQParser class modified for Ginga plugin.
class DQParser(object):
    """Class to handle parsing of DQ flags.

    **Definition Table**

    A "definition table" is an ASCII table that defines
    each DQ flag and its short and long descriptions.
    It can have optional comment line(s) for metadata,
    e.g.::

        # TELESCOPE = ANY
        # INSTRUMENT = ANY

    It must have three columns:

    1. ``DQFLAG`` contains the flag value (``uint16``).
    2. ``SHORT_DESCRIPTION`` (string).
    3. ``LONG_DESCRIPTION`` (string).

    Example file contents::

        # INSTRUMENT = HSTGENERIC
        DQFLAG SHORT_DESCRIPTION LONG_DESCRIPTION
        0      "OK"              "Good pixel"
        1      "LOST"            "Lost during compression"
        ...    ...               ...

    The table format must be readable by ``astropy.io.ascii``.

    Parameters
    ----------
    definition_file : str
        ASCII table that defines the DQ flags (see above).

    Attributes
    ----------
    tab : ``astropy.table.Table``
        Table object from given definition file.

    metadata : ``astropy.table.Table``
        Table object from file metadata.

    """
    def __init__(self, definition_file):
        self._dqcol = 'DQFLAG'
        self._sdcol = 'short'  # SHORT_DESCRIPTION
        self._ldcol = 'long'   # LONG_DESCRIPTION

        # Need to replace ~ with $HOME
        self.tab = ascii.read(
            os.path.expanduser(definition_file),
            names=(self._dqcol, self._sdcol, self._ldcol),
            converters={self._dqcol: [ascii.convert_numpy(np.uint16)],
                        self._sdcol: [ascii.convert_numpy(np.str)],
                        self._ldcol: [ascii.convert_numpy(np.str)]})

        # Another table to store metadata
        self.metadata = ascii.read(self.tab.meta['comments'], delimiter='=',
                                   format='no_header', names=['key', 'val'])

        # Ensure table has OK flag to detect good pixel
        self._okflag = 0
        if self._okflag not in self.tab[self._dqcol]:
            self.tab.add_row([self._okflag, 'OK', 'Good pixel'])

        # Sort table in ascending order
        self.tab.sort(self._dqcol)

        # Compile a list of flags
        self._valid_flags = self.tab[self._dqcol]

    def interpret_array(self, data):
        """Interpret DQ values for an array.

        .. warning::

            If the array is large and has a lot of flagged elements,
            this can be resource intensive.

        Parameters
        ----------
        data : ndarray
            DQ values.

        Returns
        -------
        dqs_by_flag : dict
            Dictionary mapping each interpreted DQ value to indices
            of affected array elements.

        """
        data = np.asarray(data, dtype=np.int)  # Ensure int array
        dqs_by_flag = {}

        def _one_flag(vf):
            dqs_by_flag[vf] = np.where((data & vf) != 0)

        # Skip good flag
        list(map(_one_flag, self._valid_flags[1:]))

        return dqs_by_flag

    def interpret_dqval(self, dqval):
        """Interpret DQ values for a single pixel.

        Parameters
        ----------
        dqval : int
            DQ value.

        Returns
        -------
        dqs : ``astropy.table.Table``
            Table object containing a list of interpreted DQ values and
            their meanings.

        """
        dqval = int(dqval)

        # Good pixel, nothing to do
        if dqval == self._okflag:
            idx = np.where(self.tab[self._dqcol] == self._okflag)

        # Find all the possible DQ flags
        else:
            idx = (dqval & self._valid_flags) != 0

        return self.tab[idx]
