from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod, abstractproperty

from csbot.plugin import Plugin

Token = namedtuple('Token', ['token_type', 'lexeme'])

class TokenType(Enum):
    UNKNOWN = 0
    
    # Scheme Names are standard variable names
    # they are a string of any non-reserved non-whitespace printable characters 
    # and are not Symbols
    NAME = 1
    
    # Scheme Symbols are the symbol \' (apostrophe) followed by a Name
    SYMBOL = 2

    NUMBER = 3

    # L_PAREN, R_PAREN are opening and closing parenthesises, respectfully
    L_PAREN = 4
    R_PAREN = 5

    # L_BRACKET, R_BRACKET are opening and closing brackets ('[' and ']'), respectfully 
    L_BRACKET = 6
    R_BRACKET = 7

    # Scheme strings are strings of printable characters enclosed in ``"``
    STRING = 8

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

def lex(s):
    '''Perform lexical analysis on the given string *s*

    If *s* is accepted, then this will return a list of *Token*s
    If *s* is not accepted, then this will raise `~ValueError`
    '''
    raise ValueError

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

class LispyEval(Plugin):
    '''Performs evaluations on given strings interpreted as a scheme/lisp input
    '''

    @Plugin.command('eval')
    def do_eval(self, e):
        '''Evaluate the data in *e* as if it were a lisp string
        Sending the reply
        '''
        e.reply(e["data"])
