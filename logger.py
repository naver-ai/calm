"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import logging

try:
    import ujson as json
except ImportError:
    import json

from util import set_logger

_TIMEZONE = 'Asia/Seoul'


class LoggerBase(object):
    """Base class for logger
    """

    def __init__(self, **kwargs):
        self.level = kwargs.get('level', logging.INFO)
        self.logger = set_logger(**kwargs)

    def log(self, msg, level=None):
        """log a message.

        Arg:
            msg (str): a message string to logging.
            level (int): a logging level of the message
        """
        raise NotImplementedError

    def log_dict(self, msg_dict, prefix='', level=None):
        """log a message from a dictionary object.

        Arg:
            msg_dict (dict): a dictionary to logging.
            prefix (str): a prefix of the message to logging.
            level (int): a logging level of the message.
        """
        raise NotImplementedError

    def report(self, msg_dict, prefix='', level=None):
        """report a message for a scalar graph.

        Arg:
            msg_dict (dict): a dictionary to logging.
                msg_dict should have property "step".
            level (int): a logging level of the message.
        """
        raise NotImplementedError

    def finalize_log(self):
        pass


class PythonLogger(LoggerBase):
    """a logger with Python ``'logging'`` library.
    """

    def log(self, msg, level=None):
        if level is None:
            level = self.level
        self.logger.log(level, msg)

    def log_dict(self, msg_dict, prefix='Report @step: ', level=None):
        if 'step' in msg_dict:
            step = msg_dict['step']
            prefix = '{}{:.2f} '.format(prefix, step)
        self.log('{}{}'.format(prefix, msg_dict), level=level)

    def report(self, msg_dict, prefix='Report @step', level=None):
        self.log_dict(msg_dict, prefix, level=level)


def load_logger(logger_type, **kwargs):
    if logger_type == 'PythonLogger':
        return PythonLogger(**kwargs)
    raise ValueError(f"Unknown logger {logger_type}")
