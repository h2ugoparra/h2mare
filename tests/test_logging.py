"""Tests for utils/logging.py."""

import logging as stdlib_logging
from unittest.mock import patch

from loguru import logger

from h2mare.utils.logging import LOG_FILE_FORMAT, _InterceptHandler, log_time


class TestVarContextColumn:
    """The file format carries a `var` column filled by logger.contextualize;
    messages outside any variable scope show '-'."""

    def test_contextualize_fills_var_column(self):
        captured: list[str] = []
        logger.configure(extra={"var": "-"})  # mirrors configure_logging
        sink_id = logger.add(captured.append, format=LOG_FILE_FORMAT)
        try:
            logger.info("outside any scope")
            with logger.contextualize(var="sst"):
                logger.info("inside sst scope")
        finally:
            logger.remove(sink_id)

        outside = next(line for line in captured if "outside any scope" in line)
        inside = next(line for line in captured if "inside sst scope" in line)
        assert "| -" in outside
        assert "| sst" in inside


class TestInterceptHandler:
    """Stdlib logging records (cdsapi, copernicusmarine, …) must flow into
    loguru so they land in the same pipeline.log file sink."""

    def _stdlib_logger_with_intercept(self, name: str) -> stdlib_logging.Logger:
        lg = stdlib_logging.getLogger(name)
        lg.setLevel(stdlib_logging.INFO)
        lg.handlers = [_InterceptHandler()]
        lg.propagate = False
        return lg

    def test_stdlib_record_reaches_loguru_sink(self):
        captured: list[str] = []
        sink_id = logger.add(captured.append, format="{level}|{message}")
        try:
            lg = self._stdlib_logger_with_intercept("test_intercept_a")
            lg.info("request accepted")
        finally:
            logger.remove(sink_id)

        assert any("INFO|request accepted" in line for line in captured)

    def test_unknown_level_falls_back_to_levelno(self):
        captured: list[str] = []
        sink_id = logger.add(captured.append, format="{message}")
        try:
            lg = self._stdlib_logger_with_intercept("test_intercept_b")
            lg.log(35, "custom level message")  # 35 has no loguru name
        finally:
            logger.remove(sink_id)

        assert any("custom level message" in line for line in captured)


class TestLogTime:
    def test_returns_function_result(self):
        @log_time
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_fast_path_logged(self):
        @log_time
        def fast():
            return "done"

        with patch("h2mare.utils.logging.time.perf_counter", side_effect=[0.0, 1.5]):
            with patch("h2mare.utils.logging.logger.info") as mock_log:
                fast()
        mock_log.assert_called_once()
        assert "secs" in mock_log.call_args[0][0]

    def test_slow_path_logged_in_minutes(self):
        @log_time
        def slow():
            return "done"

        with patch("h2mare.utils.logging.time.perf_counter", side_effect=[0.0, 125.0]):
            with patch("h2mare.utils.logging.logger.info") as mock_log:
                slow()
        mock_log.assert_called_once()
        assert "min" in mock_log.call_args[0][0]

    def test_kwargs_forwarded(self):
        @log_time
        def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"
