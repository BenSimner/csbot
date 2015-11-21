import unittest
from unittest import mock
import asyncio
import asyncio.test_utils
import os
from io import StringIO
from textwrap import dedent
import gc
import functools
from unittest import mock

from csbot.core import Bot, BotClient


def mock_reader_writer(loop):
    """Create a mock (reader, writer) pair.

    The writer is augmented with a mock write method, and ``writer.close()``
    directly causes EOF on ``reader``.
    """
    reader = asyncio.StreamReader(loop=loop)
    class MockStreamWriter(asyncio.StreamWriter):
        def close(self):
            self._reader.feed_eof()
    writer = MockStreamWriter(None, None, reader, loop)
    writer.write = mock.Mock()
    return reader, writer


def mock_client(cls, *args, **kwargs):
    """Create an instance of a mocked subclass of *cls*.

    *cls* should be :class:`IRCClient` or a subclass of it.  The event loop and
    transport are mocked, and a :meth:`reset_mock` method is added for resetting
    all mocks on the client.
    """
    class MockIRCClient(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.send_line = mock.Mock(wraps=self.send_line)

        @asyncio.coroutine
        def connect(self):
            # Connect to mock streams
            self.reader, self.writer = mock_reader_writer(self.loop)

        def reset_mock(self):
            self.send_line.reset_mock()

    return MockIRCClient(*args, **kwargs)


class IRCClientTestCase(asyncio.test_utils.TestCase):
    #: The IRCClient (sub)class to instrument
    CLIENT_CLASS = None

    def setUp(self):
        assert self.CLIENT_CLASS is not None, "no CLIENT_CLASS set on test case"
        super().setUp()
        # Create event loop and ensure everything will be using it explicitly
        self.loop = asyncio.new_event_loop()
        self.set_event_loop(self.loop)  # Disables "default" event loop!
        # Create client and make it use our event loop
        self.client = mock_client(self.CLIENT_CLASS)
        self.client.loop = self.loop
        # Connect fake stream reader/writer
        self.loop.run_until_complete(self.client.connect())

    def tearDown(self):
        # Give queued tasks a final chance to complete - borrowed from
        # StreamReaderTests in tests/test_streams.py in asyncio.
        asyncio.test_utils.run_briefly(self.loop)
        self.loop.close()
        gc.collect()
        super().tearDown()

    def patch(self, attrs, create=False):
        """Shortcut for patching attribute(s) of the client."""
        if isinstance(attrs, str):
            return mock.patch.object(self.client, attrs, create=create)
        else:
            return mock.patch.multiple(self.client, create=create,
                                       **{k: mock.DEFAULT for k in attrs})

    def receive_bytes(self, bytes):
        """Shortcut for pushing received data to the client."""
        self.client.reader.feed_data(bytes)

    def assert_bytes_sent(self, bytes):
        """Check the raw bytes that have been sent via the transport.

        Compares *bytes* to the collection of everything sent to
        ``transport.write(...)``.  Resets the mock so the next call will not
        contain what was checked by this call.
        """
        sent = b''.join(args[0] for args, _ in self.client.writer.write.call_args_list)
        self.assertEqual(sent, bytes)
        self.client.writer.write.reset_mock()

    def receive(self, lines):
        """Shortcut to push a series of lines to the client."""
        if isinstance(lines, str):
            lines = [lines]
        for l in lines:
            self.client.line_received(l)

    def assert_sent(self, lines):
        """Check that a list of (unicode) strings have been sent.

        Resets the mock so the next call will not contain what was checked by
        this call.
        """
        if isinstance(lines, str):
            lines = [lines]
        self.client.send_line.assert_has_calls([mock.call(l) for l in lines])
        self.client.send_line.reset_mock()


def run_client(f):
    """Helper for tests that require actually running the bot read loop.

    A test decorated with this function should be a coroutine, i.e. at some
    point it should ``yield`` in some way to allow the client to progress.

    >>> class TestFoo(IRCClientTestCase):
    ...     @run_client
    ...     def test_something(self):
    ...         self.receive_bytes(b":nick!user@host PRIVMSG #channel :hello\r\n")
    ...         yield
    ...         self.assert_sent('PRIVMSG #channel :what do you mean, hello?')
    """
    @functools.wraps(f)
    def new_f(self):
        # Start the client read loop
        read_loop_fut = self.loop.create_task(self.client.read_loop())
        # Run the test coroutine
        self.loop.run_until_complete(f(self))
        # Cleanly end the read loop
        self.client.reader.feed_eof()
        self.loop.run_until_complete(read_loop_fut)
    return new_f


class TempEnvVars(object):
    """A context manager for temporarily changing the values of environment
    variables."""
    def __init__(self, changes):
        self.changes = changes
        self.restore = {}

    def __enter__(self):
        for k, v in self.changes.items():
            if k in os.environ:
                self.restore[k] = os.environ[k]
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for k, v in self.changes.items():
            if k in self.restore:
                os.environ[k] = self.restore[k]
            else:
                del os.environ[k]


class BotTestCase(unittest.TestCase):
    """Common functionality for bot test case.

    A :class:`unittest.TestCase` with bot and plugin test fixtures.  Creates a
    bot from :attr:`CONFIG`, binding it to ``self.bot``, and also binding every
    plugin in :attr:`PLUGINS` to ``self.plugin``.
    """
    CONFIG = ""
    PLUGINS = []

    def setUp(self):
        """Create bot and plugin bindings."""
        # Bot and protocol stuff, suffixed with _ so they can't clash with
        # possible/likely plugin names
        self.bot_ = Bot(StringIO(dedent(self.CONFIG)))
        self.bot_.bot_setup()
        self.protocol_ = mock_client(BotClient, self.bot_)
        self.protocol_.connect()
        self.protocol_.reset_mock()

        for p in self.PLUGINS:
            setattr(self, p, self.bot_.plugins[p])

    def tearDown(self):
        """Lose references to bot and plugins."""
        self.bot_ = None
        self.protocol_ = None
        for p in self.PLUGINS:
            setattr(self, p, None)


def fixture_file(*path):
    """Get the path to a fixture file."""
    return os.path.join(os.path.dirname(__file__), 'fixtures', *path)


def read_fixture_file(*path, mode='rb'):
    """Read the contents of a fixture file."""
    with open(fixture_file(*path), mode) as f:
        return f.read()
