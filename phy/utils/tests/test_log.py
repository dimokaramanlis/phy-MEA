# -*- coding: utf-8 -*-

"""Tests of the log file."""

import logging

from .._log import _add_log_file


def test_add_log_file(tempdir):
    path = tempdir / 'phy.log'
    logger = logging.getLogger('phy.test_log')
    h1 = _add_log_file(path, logger_name='phy.test_log')
    # Adding the same file twice does nothing.
    h2 = _add_log_file(tempdir / '.' / 'phy.log', logger_name='phy.test_log')
    assert h1 is h2
    try:
        logger.debug("hello µ")
        h1.flush()
        lines = path.read_text(encoding='utf-8').splitlines()
        assert len(lines) == 1
        assert '[D]' in lines[0] and 'hello µ' in lines[0]
        # No terminal color codes in the file.
        assert '\33[' not in lines[0]
    finally:
        logger.removeHandler(h1)
        h1.close()
