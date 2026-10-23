# -*- coding: utf-8 -*-

"""Log file."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
from pathlib import Path


#------------------------------------------------------------------------------
# Log file
#------------------------------------------------------------------------------

_LOG_FMT = '%(asctime)s.%(msecs)03d [%(levelname_short)s] %(caller)s %(message)s'
_LOG_DATE_FMT = '%H:%M:%S'


class _FileFormatter(logging.Formatter):
    """Same layout as phylib's console formatter, without the terminal color codes.

    NOTE: phylib's formatter also truncates `record.levelname` in place, which affects all
    the other handlers of the record. Here the short names are stored in separate attributes.

    """
    def format(self, record):
        record.levelname_short = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        return super(_FileFormatter, self).format(record)


def _add_log_file(filename, logger_name='phy'):
    """Write the log of phy with DEBUG level to a file.

    Nothing is done if the logger already writes to that file. phylib's `_add_log_file()`
    added a new handler at each call: every line was written twice in `phy.log`, as the file
    was added both when launching the GUI and when creating the controller.

    """
    filename = str(Path(filename).resolve())
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        if (isinstance(handler, logging.FileHandler) and
                op.normcase(handler.baseFilename) == op.normcase(filename)):
            return handler
    handler = logging.FileHandler(filename, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_FileFormatter(fmt=_LOG_FMT, datefmt=_LOG_DATE_FMT))
    logger.addHandler(handler)
    return handler
