"""Tests for utils/logging_utils.py."""

from unittest.mock import patch

from h2mare.utils.logging_utils import log_time


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

        with patch(
            "h2mare.utils.logging_utils.time.perf_counter", side_effect=[0.0, 1.5]
        ):
            with patch("h2mare.utils.logging_utils.logger.info") as mock_log:
                fast()
        mock_log.assert_called_once()
        assert "secs" in mock_log.call_args[0][0]

    def test_slow_path_logged_in_minutes(self):
        @log_time
        def slow():
            return "done"

        with patch(
            "h2mare.utils.logging_utils.time.perf_counter", side_effect=[0.0, 125.0]
        ):
            with patch("h2mare.utils.logging_utils.logger.info") as mock_log:
                slow()
        mock_log.assert_called_once()
        assert "min" in mock_log.call_args[0][0]

    def test_kwargs_forwarded(self):
        @log_time
        def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"
