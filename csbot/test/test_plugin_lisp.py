from csbot.test import BotTestCase
from csbot.plugins.lisp import *

TT = TokenType

lexer_test_cases_language_symbols = [
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
    ('123.1', [TT.REAL], ['123.1']),
    ('123.12', [TT.REAL], ['123.12']),
    ('123.12i', [TT.IMAGINARY], ['123.12i']),
    ('123i', [TT.IMAGINARY], ['123i']),
]

lexer_test_cases_strings = [
    ('"a"', [TT.STRING], ['a']),
    ('""', [TT.STRING], ['']),
    (r'"\""', [TT.STRING], ['"']),
    (r'"a\"b"', [TT.STRING], ['a"b']),
    (r'"\t"', [TT.STRING], ['\t']),
    (r'"\n"', [TT.STRING], ['\n']),
]

lexer_test_cases_symbols = [
    ("'a", [TT.SYMBOL], ['a']),
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

    def _test(self, test_cases):
        for s, x, y in test_cases:
            with self.subTest(s=s):
                result = self.lisp._lex(s)
                self._match(result, x, y)

    def test_lexer_language_symbols(self):
        self._test(lexer_test_cases_language_symbols)

    def test_lexer_strings(self):
        self._test(lexer_test_cases_strings)

    def test_lexer_symbols(self):
        self._test(lexer_test_cases_symbols)

    def test_lexer_symbol_match(self):
        self._test(lexer_test_cases_match_symbols)

    def test_lexer_numbers(self):
        self._test(lexer_test_cases_numbers)

    def test_lexer_names(self):
        self._test(lexer_test_cases_names)

    def test_lexer_whitespace(self):
        self._test(lexer_test_cases_whitespace_ignore)


parser_test_cases_atoms = [
    # numbers stored as complex values
    ('5i', NumAST, [0,5]), 
    ('5', NumAST, [5,0]), 
    ('5.1', NumAST, [5.1,0]), 
    ('5.1i', NumAST, [0, 5.1]), 

    # symbols
    ("'a", SymbolAST, ['a']),

    # strings
    ('"abc"', StringAST, ['abc']),
    (r'"a\"bc"', StringAST, ['a"bc']),

    # names
    ('f', NameAST, ['f']),
]

parser_test_cases_exprs = [
    ('(f)', FuncApplicationAST, [NameAST('f')]),
    ('(f x)', FuncApplicationAST, [NameAST('f'), NameAST('x')]),
    ('(f x)', FuncApplicationAST, [NameAST('f'), NameAST('x')]),
    ('(f (x y) x)', FuncApplicationAST, [NameAST('f'), FuncApplicationAST(NameAST('x'), NameAST('y')), NameAST('x')]),
]

class TestParser(BotTestCase):
    CONFIG = """\
    [@bot]
    plugins = lisp
    """

    PLUGINS = ['lisp']

    def _test(self, cases):
        for case in cases:
            s, expected_ast, expected_children = case 
            with self.subTest(s=s):
                tks = self.lisp._lex(s)
                ast = self.lisp._parse(tks)
                self.assertEqual(ast.__class__, expected_ast)
                # coerce both into lists
                # since we're testing that the iterables contain the same values
                # not that they're contained within same iterable type
                self.assertEqual(list(ast.children), list(expected_children))

    def test_parser_atoms(self):
        self._test(parser_test_cases_atoms)

    def test_parser_exprs(self):
        self._test(parser_test_cases_exprs)
