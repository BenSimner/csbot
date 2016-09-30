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
        ('(print 3) 4', '3\t4'),
        ('(print (+ 1 1))', '2'),
        ('(list 1 2 3)', "'(1 2 3)"),
        ('(cons 1 2)', "'(1 . 2)"),
        ('(cons 1 (cons 2 3))', "'(1 2 . 3)"),
        #('(cons 1 (list))', "'(1)"),
        #('(car (cons 1 2))', '1'),
        #('(cdr (cons 1 2))', '2'),
        ('#t', '#t'),
        ('(if #t 1 2)', '1'),
        ('3', '3'),
        ('"hello"', '"hello"'),
        ('"hello world"', '"hello world"'),
        ('(+ 1 2 3)', '6'),
        ('1 2', '2'),
        ('(+ 1 2 3) (+ 2 3)', '5'),
        ('(lambda (x) (+ 1 x))', '<lambda>'),
        ('[+ 1 2]', '3'),
        ('((lambda (x y) (+ x y)) 3 4)', '7'),
        ('((lambda (x) (+ 1 x)) 1)', '2'),
        ('(define (f x) x) (f 3)', '3'),
        ('(- 5 1)', '4'),
        ('(= 1 1)', '#t'),
        ('(define (f n) (if (= n 0) 1 [f n])) (f 0)', '1'),
    ]

    INTEGRATION_TESTS = [ ('(cons 1 2)', "'(1 . 2)")]
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
