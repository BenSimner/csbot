from csbot.test import BotTestCase


class TestLispPluginIntegration(BotTestCase):
    CONFIG = '''\
    [@bot]
    plugins = lisp
    '''

    PLUGINS = ['lisp']

    INTEGRATION_TESTS = [
        ('(print "hello")', '"hello"'),
        ('(print 3)', '3'),
        ('(print #t)', '#t'),
        ('#t', '#t'),
        ('(if #t 1 2)', '1'),
        ('3', '3'),
        ('"hello"', '"hello"'),
        ('"hello world"', '"hello world"'),
    ]

    def eval_equals(self, s, e):
        return self.assertEqual(self.lisp.lisp_eval(s), e)

    def test_integrations(self):
        for s, e in self.INTEGRATION_TESTS:
            with self.subTest('integration test: `{}` evals to `{}`'.format(s, e)):
                self.eval_equals(s, e)

class TestLispPluginParser(BotTestCase):
    CONFIG = '''\
    [@bot]
    plugins = lisp
    '''

    PLUGINS = ['lisp']
