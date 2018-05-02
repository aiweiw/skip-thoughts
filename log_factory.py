#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging


class FinalLogger(object):
    """Logger."""

    def __init__(self, log_path='final_logger.log', file_log_level=logging.DEBUG, stream_log_level=logging.DEBUG):
        self._logger = logging.getLogger(log_path)
        self._logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(stream_log_level)
        # file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(file_log_level)
        # add handler
        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def get_logger(self):
        return self._logger
