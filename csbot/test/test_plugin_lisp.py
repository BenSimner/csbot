from csbot.test import BotTestCase
from csbot.plugins.lisp import TokenType, Token

TT = TokenType

lexer_test_cases_symbols = [
    ('(', [TT.L_PAREN],   ['(']),
    (')', [TT.R_PAREN],   [')']),
    ('[', [TT.L_BRACKET], ['[']),
    (']', [TT.R_BRACKET], [']']),
]

lexer_test_cases_match_symbols = [
    ('()', [TT.L_PAREN, TT.R_PAREN], ['(', ')']),
    ('[]', [TT.L_BRACKET, TT.R_BRACKET], ['[', ']']),
    ('[123]', [TT.L_BRACKET, TT.NUMBER, TT.R_BRACKET], ['[', '123', ']']),
]

lexer_test_cases_numbers = [
    ('1', [TT.NUMBER], ['1']),
    ('123', [TT.NUMBER], ['123']),
    ('123.1', [TT.NUMBER], ['123.1']),
    ('123.12', [TT.NUMBER], ['123.12']),
    ('123.12i', [TT.IMAGINARY], ['123.12i']),
]

lexer_test_cases_names = [
    ('x', [TT.NAME], ['x']),
    ('x\'', [TT.NAME], ['x\'']),
    ('0a', [TT.NAME], ['0a']),
]

lexer_test_cases_whitespace_ignore = [
    ('x x', [TT.NAME, TT.NAME], ['x', 'x']),
]

class TestLexer(BotTestCase):
    CONFIG = """\
    [@bot]
    plugins = lisp
    """

    PLUGINS = ['lisp']

    def _match(self, result, token_types, lexemes):
        tokens = map(lambda x: Token(*x), zip(token_types, lexemes))

        result = list(result)
        tokens = list(tokens)

        self.assertEqual(result, tokens)


    def test_lexer_symbols(self):
        for s, x, y in lexer_test_cases_symbols:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

    def test_lexer_symbol_match(self):
        for s, x, y in lexer_test_cases_match_symbols:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

    def test_lexer_numbers(self):
        for s, x, y in lexer_test_cases_numbers:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

    def test_lexer_names(self):
        for s, x, y in lexer_test_cases_names:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

    def test_lexer_whitespace(self):
        for s, x, y in lexer_test_cases_whitespace_ignore:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

