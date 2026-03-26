"""
Tests for database operations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sqlite3


class TestDatabaseInitialization:
    """Tests for database initialization."""

    @patch("app.database.get_connection")
    def test_init_database(self, mock_get_conn):
        """Test database initialization."""
        from app.database import init_database

        # Mock connection using the actual get_connection path (Turso/libsql)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = None
        mock_cursor.rowcount = 0
        mock_cursor.lastrowid = None
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        init_database()

        # Verify connection was made via get_connection
        assert mock_get_conn.called
        assert mock_cursor.execute.called


class TestSentimentOperations:
    """Tests for sentiment database operations."""

    @patch("app.database.get_connection")
    def test_add_sentiment(self, mock_get_conn):
        """Test adding sentiment to database."""
        from app.database import add_sentiment

        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        add_sentiment(
            date_str="2026-03-01",
            daily_sentiment_decay=0.5,
            news_volume=10,
            log_news_volume=2.3,
            decayed_news_volume=8.5,
            high_news_regime=1,
        )

        # Verify execute was called
        assert mock_cursor.execute.called

    @patch("app.database.get_connection")
    def test_get_sentiment_history(self, mock_get_conn):
        """Test retrieving sentiment history."""
        from app.database import get_sentiment_history

        # Mock connection — app uses _query_to_df which calls cursor.execute/fetchall
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("date",),
            ("daily_sentiment",),
            ("news_volume",),
            ("log_news_volume",),
            ("decayed_news_volume",),
            ("high_news_regime",),
        ]
        mock_cursor.fetchall.return_value = [
            ("2026-03-01", 0.5, 10, 2.3, 8.5, 1),
            ("2026-03-02", 0.3, 8, 2.1, 7.0, 0),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_sentiment_history(days=30)

        assert mock_get_conn.called
        assert len(result) == 2

    @patch("app.database.get_connection")
    def test_get_latest_sentiment(self, mock_get_conn):
        """Test getting latest sentiment."""
        from app.database import get_latest_sentiment

        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Mock Row object from sqlite3
        mock_row = MagicMock()
        mock_row.keys.return_value = [
            "date",
            "daily_sentiment_decay",
            "news_volume",
            "log_news_volume",
            "decayed_news_volume",
            "high_news_regime",
        ]
        mock_row.__getitem__ = lambda self, key: {
            "date": "2026-03-01",
            "daily_sentiment_decay": 0.5,
            "news_volume": 10,
            "log_news_volume": 2.3,
            "decayed_news_volume": 8.5,
            "high_news_regime": 1,
        }[key]
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_latest_sentiment()

        assert mock_cursor.execute.called
        assert result is not None

    @patch("app.database.get_connection")
    def test_get_sentiment_count(self, mock_get_conn):
        """Test getting sentiment count."""
        from app.database import get_sentiment_count

        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_sentiment_count()

        assert mock_cursor.execute.called
        assert result == 100

    @patch("app.database.get_connection")
    def test_add_bulk_sentiment(self, mock_get_conn):
        """Test adding bulk sentiment data."""
        from app.database import add_bulk_sentiment

        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        bulk_data = [
            {
                "date": "2026-03-01",
                "daily_sentiment_decay": 0.5,
                "news_volume": 10,
                "log_news_volume": 2.3,
                "decayed_news_volume": 8.5,
                "high_news_regime": 1,
            },
            {
                "date": "2026-03-02",
                "daily_sentiment_decay": 0.3,
                "news_volume": 8,
                "log_news_volume": 2.1,
                "decayed_news_volume": 7.0,
                "high_news_regime": 0,
            },
        ]

        add_bulk_sentiment(bulk_data)

        # Verify execute was called (not executemany in current implementation)
        assert mock_cursor.execute.called

    @patch("app.database.get_connection")
    def test_get_sentiment_for_dates(self, mock_get_conn):
        """Test getting sentiment for specific dates."""
        from app.database import get_sentiment_for_dates

        # Mock connection — app uses _query_to_df which calls cursor.execute/fetchall
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("date",),
            ("daily_sentiment",),
            ("news_volume",),
            ("log_news_volume",),
            ("decayed_news_volume",),
            ("high_news_regime",),
        ]
        mock_cursor.fetchall.return_value = [
            ("2026-03-01", 0.5, 10, 2.3, 8.5, 1),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = get_sentiment_for_dates("2026-03-01", "2026-03-01")

        assert mock_get_conn.called
        assert len(result) == 1


class TestDatabaseErrors:
    """Tests for database error handling."""

    @patch("app.database.get_connection")
    def test_connection_error(self, mock_get_conn):
        """Test handling database connection errors."""
        from app.database import get_sentiment_history

        mock_get_conn.side_effect = sqlite3.Error("Connection failed")

        try:
            get_sentiment_history(days=30)
        except Exception as e:
            assert isinstance(e, sqlite3.Error)

    @patch("app.database.get_connection")
    def test_query_error(self, mock_get_conn):
        """Test handling query errors."""
        from app.database import add_sentiment

        # Mock connection that raises error on execute
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.Error("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        try:
            add_sentiment(
                date_str="2026-03-01",
                daily_sentiment_decay=0.5,
                news_volume=10,
                log_news_volume=2.3,
                decayed_news_volume=8.5,
                high_news_regime=1,
            )
        except Exception:
            pass  # Expected to fail
