from string import whitespace
from enum import Enum
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod, abstractproperty

from csbot.plugin import Plugin

Token = namedtuple('Token', ['token_type', 'lexeme'])

class TokenType(Enum):
    UNKNOWN = 0
    
    # Scheme Names are standard variable names 
    # they are a string of any non-reserved non-whitespace printable characters 
    NAME = 1
    
    # An apostrophe can either appear as a part of the name, 
    # but when it appears at the _start_ of an expr, it is evaluated as a symbol
    # and is used to build up scheme symbols (quoted expressions)
    APOSTROPHE = 2

    # Scheme strings are strings of printable characters enclosed in ``"``
    STRING = 3

    # L_PAREN, R_PAREN are opening and closing parenthesises, respectfully
    L_PAREN = 4
    R_PAREN = 5

    # L_BRACKET, R_BRACKET are opening and closing brackets ('[' and ']'), respectfully 
    L_BRACKET = 6
    R_BRACKET = 7

    NUMBER = 8
    REAL = 9
    IMAGINARY = 10

SYMBOL_LUT = {
    '(': TokenType.L_PAREN,
    ')': TokenType.R_PAREN,
    '[': TokenType.L_BRACKET,
    ']': TokenType.R_BRACKET
}

# Symbols do not require whitespace between them
SYMBOLS = ('(', ')', '[', ']')

class AST(ABC):
    '''The abstract-syntax-tree base for a scheme-like language

    An AST can have a list of child :class:`AST`s
    '''

    def ofType(self, cls):
        '''Helper function to test whether two :class:`AST` instances are infact instances of the the same class
        '''
        return type(self) == cls

    def __eq__(self, other):
        '''Equality is defined over the children
        '''
        return self.children == other.children and type(other) == type(self)

    @abstractproperty
    def children(self):
        '''Returns an iterable of children :class:`AST`s associated with this class
        '''

    def __repr__(self):
        return '<{}: children={}>'.format(self.__class__.__name__, list(map(repr,self.children)))

class NumAST(AST):
    def __init__(self, real, imaginary=0):
        self._real = real
        self._imaginary = imaginary

    @property
    def children(self):
        '''For atomic values (numbers, symbols, strings...)
        the values returned by ``children`` are python built-ins and not :class:`AST`s
        '''

        return [self._real, self._imaginary]

def make_ast(name, arg_names=[]):
    class _AST(AST):
        def __init__(self,*args):
            self._children = args
        @property
        def children(self):
            return self._children

    _AST.__name__ = name
    _AST.__qualname__ = name

    doc='''Arguments:
 - {}
    '''.format('\n - '.join(arg_names))
    _AST.__doc__ = doc

    return _AST

NameAST = make_ast('Name', ['name'])
SymbolAST = make_ast('Symbol', ['expr'])
StringAST = make_ast('String', ['string'])
FuncApplicationAST = make_ast('FuncApply', ['funcName', '*args'])


def build_lexchar_dict(**words):
    '''Given a map of strings->``TokenType`` *words* build a tree from a dict allowing easy checking of a word.'''
    matches = LexCharDict()

    for word,tok in words.items():
        match = matches
        for letter in word:
            match = match[letter]
        match._tok=tok # overwrite token type for this match

    return matches


class LexCharDict(defaultdict):
    '''A ``defaultdict`` which maps single characters to other defaultdicts
        this behaves as a tree, allowing lookup of tokentypes of lexeme's incrementally in a loop
    
    This class utilizes ``defaultdict`` as a FSA to accept regular languages and determine
    the :class:`TokenType` of the given input string.
    '''

    def __init__(self, char=None, tok=None):
        super().__init__()
        self._char = char
        self._tok = tok
        
    def __missing__(self, k):
        d = self.__class__(k, self._tok)

        # handle strings (enclosed in ``"``)
        if k == '"':
            if self._tok is None:
                d._char = ''
                d._tok = 'unmatched_string'
            if self._tok is 'unmatched_string':
                d._char = ''
                d._tok = TokenType.STRING
        elif k == '\\' and self._tok is 'unmatched_string':
            # handle escaped characters
            # by automatically adding escaped characters to d
            d['"'] = self.__class__('"', 'unmatched_string')
            c = d['"']
            c._char = '"'
            c = d['n']
            c._char = '\n'
            c = d['t']
            c._char = '\t'
            d['\\'] = self.__class__('\\', None)
            c = d['\\']
            c._char = '\\'
            c._tok = self._tok
            d._char = ''

        # handle symbols
        elif k == '\'':
            if self._tok is None:
                d._char = '\''
                d._tok = TokenType.APOSTROPHE

        # handle numbers
        elif self._tok in (None, '.', TokenType.REAL, TokenType.NUMBER):
            try:
                int(k)
                if d._tok is None:
                    d._tok = TokenType.NUMBER
                elif d._tok is '.':
                    d._tok = TokenType.REAL
            except:
                if k == 'i' and self._tok in (TokenType.NUMBER, TokenType.REAL):
                    d._tok = TokenType.IMAGINARY
                elif self._tok == TokenType.NUMBER:
                    if k == '.':
                        d._tok = '.'
                    else:
                        d._tok = TokenType.NAME
                else:
                    d._tok = TokenType.NAME
        elif self._tok == TokenType.IMAGINARY:
            d._tok = TokenType.NAME

        self[k] = d
        return d

class LexemeDict(defaultdict):
    '''A ``defaultdict`` which maps lexeme's (strings) to their tokens
    '''
    def __init__(self, matches):
        super().__init__()
        self._matches = matches

    def check_type(self, word, tok):
        if type(tok) == str:
            raise ValueError('LexerError: Unexpected `{}` when lexing `{}`'.format(tok, word))

    def __missing__(self, lex):
        match = self._matches

        word = ''
        for letter in lex:
            match = match[letter]
            word += match._char


        self.check_type(lex, match._tok)
        self[lex] = Token(match._tok, word)
        return self[lex]

class ProgramDict(defaultdict):
    '''A ``defaultdict`` which maps program strings to their tokens
    '''

    def __init__(self, symbol_list, lexeme_dict):
        super().__init__()
        self._symbols = symbol_list
        self._dict = lexeme_dict

    def __missing__(self, s):
        lex_dict = self._dict
        symbols = self._symbols

        def _gen():
            word = ''
            string = False
            escape = False
            for letter in s:
                # logic for handling strings
                # this is the only place where whitespace doesn't break up tokens
                if letter == '"':
                    if not escape:
                        string = not string
                    word += letter
                    continue
                elif string:
                    word += letter
                    escape = False
                    if letter == '\\':
                        escape = True
                    continue

                if letter in whitespace or letter in symbols:
                    if word:
                        yield lex_dict[word]
                    if letter in symbols:
                        yield lex_dict[letter]
                    word = ''
                else:
                    # deal with symbols
                    if not word and letter is '\'':
                        yield lex_dict['\'']
                        continue
                    word += letter

            # catch final token
            if word:
                yield lex_dict[word]

        self[s] = _gen()
        return self[s]

def lex(program_dict, s):
    '''Perform lexical analysis on the given string *s* using a match dictionary *program_dict* 
    (see :class:`ProgramDict`)

    If *s* is accepted, then this will return a generator of *Token*s
    If *s* is not accepted, then this will raise `~ValueError`
    '''

    return program_dict[s]

TOKENS = None
STACK = None
_CURRENT_TOKEN = None

def pop():
    return STACK.pop()

def push(p):
    STACK.append(p)

## Lookahead Functions

def _lookahead_token():
    global _CURRENT_TOKEN
    if _CURRENT_TOKEN is None:
        _CURRENT_TOKEN = next(TOKENS)

    return _CURRENT_TOKEN

def lookahead():
    return _lookahead_token().token_type

def lexeme():
    return _lookahead_token().lexeme

def accept(token_type):
    if lookahead() != token_type:
        raise ValueError('ParseError: Expected {}, got {}'.format(token_type, lookahead()))

    global _CURRENT_TOKEN
    _CURRENT_TOKEN = None

def lookahead_open():
    return lookahead() in (TokenType.L_PAREN, TokenType.L_BRACKET)

def accept_open():
    '''Accept an opening bracket/paren
    and return the expected closing bracket/paren
    '''
    if lookahead() == TokenType.L_PAREN:
        accept(TokenType.L_PAREN)
        return TokenType.R_PAREN

    accept(TokenType.L_BRACKET)
    return TokenType.R_BRACKET


def _parse_expr():
    '''Parse a scheme expression
    '''
    if lookahead_open():
        close = accept_open()
        _parse_func_call(close)
    else:
        _parse_atom()

def _parse_atom():
    if lookahead() == TokenType.NUMBER:
        num = int(lexeme())
        push(NumAST(num))
    elif lookahead() == TokenType.REAL:
        num = float(lexeme())
        push(NumAST(num))
    elif lookahead() == TokenType.IMAGINARY:
        num = float(lexeme()[:-1])  # strip the i
        push(NumAST(0, num))
    elif lookahead() == TokenType.STRING:
        s = lexeme()
        push(StringAST(s))
    elif lookahead() == TokenType.APOSTROPHE:
        accept(TokenType.APOSTROPHE)
        _parse_expr()
        push(SymbolAST(pop()))
        return
    else:
        # anything left *must* be a name
        # else it's a parse error
        _parse_name()
        return

    accept(lookahead())

def _parse_name():
    s = lexeme()
    accept(TokenType.NAME)
    push(NameAST(s))

def _parse_func_call(close_block):
    '''Parses a function call from the first expr in the parens
    '''
    if lookahead() == close_block:
        # no body
        # whilst this is semantically meaningless
        # it can be valid when quoted
        push(FuncApplicationAST([]))
        accept(close_block)
        return

    _parse_expr()
    func = pop()

    exprs = []
    while lookahead() != close_block:
        _parse_expr()
        expr = pop()
        exprs.append(expr)

    push(FuncApplicationAST(func, *exprs))
    accept(close_block)

def parse(ts):
    '''Performs syntactical analysis on the given iterable of tokens *ts*

    If *ts* is a valid iterable of tokens, this returns an :class:`AST`.
    If *ts* is invalid, this raises `~ValueError`
    '''
    global TOKENS,STACK
    TOKENS = ts
    STACK = []
    _parse_expr()
    return STACK.pop()

def eval(ast):
    '''Does semantical evaluation on some given :class:`AST` *ast*
    Returning the python object which best represents the output
    
    If the interpreter for any reason cannot determine semantics, this raies `~ValueError`
    '''

    raise ValueError

class Lisp(Plugin):
    '''Performs evaluations on given strings interpreted as a scheme/lisp input
    '''
    def _build_program_dict(self):
        matches = build_lexchar_dict(**SYMBOL_LUT)
        symbols = SYMBOLS

        lexeme_dict = LexemeDict(matches)
        program_dict = ProgramDict(symbols, lexeme_dict)
        return program_dict

    def _lex(self, s):
        program_dict = self._build_program_dict()
        return lex(program_dict, s)

    def _parse(self, tks):
        return parse(tks)

    def _eval(self, s):
        raise NotImplementedError

    @Plugin.command('eval')
    def do_eval(self, e):
        '''Evaluate the data in *e* as if it were a lisp string
        Sending the reply
        '''

        raise NotImplementedError
