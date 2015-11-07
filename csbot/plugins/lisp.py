from string import whitespace
from enum import Enum
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod, abstractproperty

from csbot.plugin import Plugin

Token = namedtuple('Token', ['token_type', 'lexeme'])
MATCHES = None
'''String matching tree'''

class TokenType(Enum):
    UNKNOWN = 0
    
    # Scheme Names are standard variable names
    # they are a string of any non-reserved non-whitespace printable characters 
    # and are not Symbols
    NAME = 1
    
    # Scheme Symbols are the symbol \' (apostrophe) followed by a Name
    SYMBOL = 2

    # Scheme strings are strings of printable characters enclosed in ``"``
    STRING = 3

    # L_PAREN, R_PAREN are opening and closing parenthesises, respectfully
    L_PAREN = 4
    R_PAREN = 5

    # L_BRACKET, R_BRACKET are opening and closing brackets ('[' and ']'), respectfully 
    L_BRACKET = 6
    R_BRACKET = 7

    NUMBER = 8
    IMAGINARY = 9

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
    
    @abstractproperty
    def children(self):
        '''Returns an iterable of children :class:`AST`s associated with this class
        '''

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
    the :class:``TokenType`` of the given input string.
    '''

    def __init__(self, tok=None):
        super().__init__()
        self._tok = tok
        
    def __missing__(self, k):
        d = self.__class__(self._tok)

        if self._tok in (None, '.', TokenType.NUMBER):
            try:
                int(k)
                if d._tok in (None, '.'):
                    d._tok = TokenType.NUMBER
            except:
                if k == 'i':
                    d._tok = TokenType.IMAGINARY
                elif k == '.':
                    d._tok = '.'
                else:
                    d._tok = TokenType.NAME
        elif self._tok == TokenType.IMAGINARY:
            d._tok = TokenType.NAME

        self[k] = d
        return d

class LexemeDict(defaultdict):
    '''A ``defaultdict`` which maps lexeme's (strings) to their token types
    '''
    def __init__(self, matches):
        super().__init__()
        self._matches = matches

    def check_type(self, tok):
        if type(tok) == str:
            raise ValueError('LexerError: Unexpected `{}`'.format(tok))

    def __missing__(self, lex):
        match = self._matches

        for letter in lex:
            match = match[letter]

        self.check_type(match._tok)

        self[lex] = match._tok
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
            for letter in s:
                if letter in whitespace or letter in symbols:
                    if word:
                        tok_type = lex_dict[word]
                        yield Token(tok_type, word)
                    if letter in symbols:
                        yield Token(lex_dict[letter], letter)
                    word = ''
                else:
                    word += letter

        self[s] = _gen()
        return self[s]

def lex(program_dict, s):
    '''Perform lexical analysis on the given string *s* using a match dictionary *program_dict* 
    (see :class:``ProgramDict``)

    If *s* is accepted, then this will return a generator of *Token*s
    If *s* is not accepted, then this will raise `~ValueError`
    '''

    return program_dict[s]

def parse(ts):
    '''Performs syntactical analysis on the given iterable of tokens *ts*

    If *ts* is a valid iterable of tokens, this returns an :class:`AST`.
    If *ts* is invalid, this raises `~ValueError`
    '''

    raise ValueError

def eval(ast):
    '''Does semantical evaluation on some given :class:`AST` *ast*
    Returning the python object which best represents the output
    
    If the interpreter for any reason cannot determine semantics, this raies `~ValueError`
    '''

    raise ValueError

class LispEval(Plugin):
    '''Performs evaluations on given strings interpreted as a scheme/lisp input
    '''

    def _eval(self, s):
        matches = build_lexchar_dict(**SYMBOL_LUT)
        symbols = SYMBOLS

        lexeme_dict = LexemeDict(matches)
        program_dict = ProgramDict(symbols, lexeme_dict)
        tk_gen = lex(program_dict, s)
        
        raise NotImplementedError

    @Plugin.command('eval')
    def do_eval(self, e):
        '''Evaluate the data in *e* as if it were a lisp string
        Sending the reply
        '''

        raise NotImplementedError
