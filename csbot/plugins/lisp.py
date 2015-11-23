# lisp.py - Lisp plugin for csbot/
# author: Ben Simner 
'''
The most over-engineered implementation of a lisp-like language i could over-engineer.
'''

from string import whitespace
from enum import Enum
from collections import namedtuple, defaultdict
from functools import partial, reduce
from abc import ABC, abstractmethod, abstractproperty
from math import log, exp

from csbot.plugin import Plugin

Token = namedtuple('Token', ['token_type', 'lexeme'])

class TokenType(Enum):
    UNKNOWN = 0
    
    # Lisp Names are standard variable names 
    # they are a string of any non-reserved non-whitespace printable characters 
    NAME = 1
    
    # An apostrophe can either appear as a part of the name, 
    # but when it appears at the _start_ of an expr, it is evaluated as a symbol
    # and is used to build up scheme symbols (quoted expressions)
    APOSTROPHE = 2

    # Lisp strings are strings of printable characters enclosed in ``"``
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

    # Literal booleans
    BOOL_TRUE = 11
    BOOL_FALSE = 12

SYMBOL_LUT = {
    '(': TokenType.L_PAREN,
    ')': TokenType.R_PAREN,
    '[': TokenType.L_BRACKET,
    ']': TokenType.R_BRACKET,
    '#t': TokenType.BOOL_TRUE,
    '#f': TokenType.BOOL_FALSE,
    '\'': TokenType.APOSTROPHE,
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
        return type(other) == type(self) and self.children == other.children

    @property
    def value(self):
        '''A hook for LispValue.value
        If a :class:`LispValue` has an :class:`AST` embedded in it 
        finding the value of that :class:`LispValue` should return just the :class:`AST`
        '''
        return self

    @abstractproperty
    def children(self):
        '''Returns an iterable of children :class:`AST`s associated with this class
        '''

    def __repr__(self):
        xs = []
        for x in self.children:
            if isinstance(x, AST):
                xs.append(repr(x))
            else:
                xs.append(str(x))
        return '<{}: children=[{}]>'.format(self.__class__.__name__, ','.join(xs))

    def pretty_out(self):
        raise NotImplementedError

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

    def pretty_out(self):
        if self._imaginary:
            return '{}+{}i'.format(repr(self._real), repr(self._imaginary))
        return repr(self._real)

class BaseAST(AST):
    def __init__(self,*args):
        self._children = args

    @property
    def children(self):
        return self._children

class NameAST(BaseAST):
    def pretty_out(self):
        name, = self.children
        return str(name)

BoolAST = type('BoolAST', (BaseAST,), {})
StringAST = type('StringAST', (BaseAST,), {})
FuncApplicationAST = type('FuncApplicationAST', (BaseAST,), {})

class SymbolAST(BaseAST):
    def pretty_out(self):
        ast, = self.children
        return ast.pretty_out()

class QuotedListAST(BaseAST):
    def pretty_out(self):
        ast, = self.children
        return '({})'.format(' '.join(map(repr, ast)))

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
            c = d['"'] = self.__class__('"', 'unmatched_string')
            c._char = '"'
            c = d['n']; c._char = '\n'
            c = d['t']; c._char = '\t'
            c = d['\\'] = self.__class__('\\', None)
            c._char = '\\'; c._tok = self._tok; d._char = ''

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
        self._missing_dict = {}

    def __missing__(self, s):
        lex_dict = self._dict
        symbols = self._symbols

        # actually we have calculated this before,
        # we just have the generator somewhere else
        if s in self._missing_dict:
            return self._missing_dict[s]()

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

        # return our token generator
        self._missing_dict[s] = _gen
        return _gen()

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
        try:
            _CURRENT_TOKEN = next(TOKENS)
        except StopIteration:
            raise ValueError('ParseError: Unexpectedly reached end of stream')


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


# take an <expr> to be quoted
# and return the (quote <expr>) version of it.
# this needs to be an AST so it can be ``dequoteify``'d and evaluated
# literals are atomic values
#  i.e. values that evaluate to themselves
_literals = (StringAST, NumAST, BoolAST)
def quoteify(ast):
    if any(map(lambda x: ast.ofType(x), _literals)):
        return ast
    elif ast.ofType(QuotedListAST):
        return ast
    elif ast.ofType(FuncApplicationAST):
        xs = ast.children
        return QuotedListAST(list(map(quoteify,xs)))
    else:
        return SymbolAST(ast)

def dequoteify(ast):
    '''Inverse of ``quoteify``
    '''
    if any(map(lambda x: ast.ofType(x), _literals)):
        return ast
    elif ast.ofType(QuotedListAST):
        xs = ast.children[0]
        return FuncApplicationAST(*map(dequoteify, xs))
    elif ast.ofType(SymbolAST):
        symb = ast.children[0]
        return symb
    raise ValueError('idk?')

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
        # TODO:
        # quoted literals are literals
        # quoted list is list of quotes
        # eval on above should coerce into ASTs
        accept(TokenType.APOSTROPHE)
        _parse_expr()
        ast = pop()
        push(quoteify(ast))
        return
    elif lookahead() == TokenType.BOOL_TRUE:
        push(BoolAST(True))
    elif lookahead() == TokenType.BOOL_FALSE:
        push(BoolAST(False))
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
        push(QuotedListAST([]))
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
    global TOKENS, STACK, _CURRENT_TOKEN
    TOKENS = ts
    STACK = []
    _CURRENT_TOKEN = None
    _parse_expr()
    return STACK.pop()

class LispValue(ABC):
    def __eq__(self, other):
        return self.value == other.value and type(self) == type(other)

    @abstractproperty
    def value(self):
        '''Extract the python value from this LispValue
        '''

    @abstractproperty
    def children(self):
        '''Extract the child LispValue's from this one
        '''

    @abstractproperty
    def pretty_out(self):
        '''Generates pretty output'''

    def __repr__(self):
        xs = []
        for x in self.children:
            if isinstance(x, LispValue):
                xs.append(repr(x))
            else:
                xs.append(str(x))
        return '<{}: children=[{}]>'.format(self.__class__.__name__, ','.join(xs))

class LispNum(LispValue):
    def __init__(self, num):
        self._num = num

    @property
    def value(self):
        return self._num

    @property
    def children(self):
        return ()

    def pretty_out(self):
        # return a lisp-lookalike value
        if self.value.imag:
            if self.value.real:
                return '{}+{}i'.format(self.value.real, self.value.imag)
            else:
                return '{}i'.format(self.value.imag)

        return repr(self.value.real)

class LispString(LispValue):
    def __init__(self, s):
        self._s = s

    @property
    def value(self):
        return self._s

    @property
    def children(self):
        return ()

    def pretty_out(self):
        return '"{}"'.format(self._s)

class LispBool(LispValue):
    def __init__(self, b):
        self._b = b

    @property
    def value(self):
        return self._b

    @property
    def children(self):
        return ()

    def pretty_out(self):
        if self._b:
            return '#t'
        return '#f'

class LispSymbol(LispValue):
    '''A symbol is just a lisp expression
    that hasn't been evaluated yet
    '''

    def __init__(self, ast):
        self._ast = ast

    @property
    def value(self):
        return self._ast

    @property
    def children(self):
        return ()

    def pretty_out(self):
        return self._ast.pretty_out()

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __repr__(self):
        return '<LispSymbol: {}>'.format(repr(self._ast))

class LispPair(LispValue):
    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def value(self):
        return (self._lhs.value, self._rhs.value)

    @property
    def children(self):
        return (self._lhs, self._rhs)

    @property
    def value_list(self):
        '''Gets the value of this LispPair as an iterable
        if the last value is LispEmptyList, will return a list
        otherwise will return a tuple
        '''
        lhs, rhs = self.children
        values = [lhs]
        if type(rhs) == LispEmptyList:
            return values

        while True:
            lhs, rhs = rhs.children
            values.append(lhs)
            if type(rhs) == LispEmptyList:
                return values
            elif type(rhs) != LispPair:
                return tuple(values)
            else:
                values += rhs.value_list
                return values

    def pretty_out(self):
        lhs, rhs = self.children
        sb = '(' + lhs.pretty_out()

        if type(rhs) == LispPair:
            while True:
                lhs, rhs = rhs.children
                sb += ' '
                sb += lhs.pretty_out()
                if type(rhs) == LispEmptyList:
                    sb += ')'
                    break
                elif type(rhs) != LispPair:
                    sb += ' . '
                    sb += rhs.pretty_out()
                    sb += ')'
                    break
        elif type(rhs) == LispEmptyList:
            sb += ')'
        else:
            sb += ' . {})'.format(rhs.pretty_out())

        return sb

class LispEmptyList(LispPair):
    def __init__(self):  
        pass

    @property
    def value(self):
        return ()

    @property
    def children(self):
        return ()

    @property
    def value_list(self):
        print('value_list!')
        return []

    def pretty_out(self):
        return '()'

class LispProc(LispValue):
    def __init__(self, argNames, bodyAST, env):
        self._args = argNames
        self._body = bodyAST
        self._env = env

    @property
    def value(self):
        return (self._args, self._body, self._env)

    def __call__(self, *args):
        '''Calling a LispProc is equivalent to calling the procedure it represents'''
        params, body, env = self.value
        env = dict(env)
        for param, arg in zip(params, args):
            name, = param.children
            env[name] = arg
        return _eval_expr(env, body)

    @property
    def children(self):
        return (self._args, self._body)

    def pretty_out(self):
        return '<procedure: {}>'.format(repr(self.children[1]))

    def __repr__(self):
        return '<procedure: {}>'.format(repr(self.children[1]))

def _eval_expr(env, expr):
    if expr.ofType(FuncApplicationAST):
        return _eval_func_apply(env, expr)
    else:
        return _eval_atom(env, expr)

def _eval_func_apply(env, expr):
    if expr.children is []:
        raise ValueError('EvalError: Found no body')

    f, *args = expr.children

    # check for special forms
    if f.ofType(NameAST):
        name, = f.children
        if name == 'if':
            return _eval_if(env, args)
        elif name == 'lambda':
            # (lambda (x y z) body)
            params = args[0].children
            body = args[1]
            return LispProc(params, body, env)
        elif name == 'let':
            bindings = args[0].children
            body = args[1]
            return _eval_let(env, bindings, body)
        elif name == 'let*':
            bindings = args[0].children
            body = args[1]
            return _eval_let_star(env, bindings, body)
        elif name == 'letrec':
            bindings = args[0].children
            body = args[1]
            return _eval_letrec(env, bindings, body)

    f_value = _eval_expr(env, f)
    # call-by-value
    arg_values = map(partial(_eval_expr, env), args)
    return f_value(*arg_values)

def _eval_if(env, exprs):
    cond, if_true, if_false = exprs
    b = _eval_expr(env, cond)
    if b.value:
        return _eval_expr(env, if_true)
    else:
        return _eval_expr(env, if_false)

def _eval_let(env, bindings, body):
    new_env = dict(env)
    for binding in bindings:
        if not binding.ofType(FuncApplicationAST):
            raise ValueError('EvalError: malformed `let` expression.')
            
        name, value_ast = binding.children
        name, = name.children
        new_env[name] = _eval_expr(env, value_ast)

    return _eval_expr(new_env, body)

def _eval_letrec(env, bindings, body):
    new_env = dict(env)
    names = []
    for binding in bindings:
        if not binding.ofType(FuncApplicationAST):
            raise ValueError('EvalError: malformed `letrec` expression.')
            
        name, value_ast = binding.children
        name, = name.children
        new_env[name] = _eval_expr(env, value_ast)
        names.append(name)

    for name in names:
        v = new_env[name]
        if isinstance(v, LispProc):
            # update each procedures' env with each name
            for n in names:
                v._env[n] = new_env[n]

    return _eval_expr(new_env, body)

def _eval_let_star(env, bindings, body):
    # create a copy
    new_env = dict(env)
    for binding in bindings:
        if not binding.ofType(FuncApplicationAST):
            raise ValueError('EvalError: malformed `let*` expression.')
            
        name, value_ast = binding.children
        name, = name.children
        new_env[name] = _eval_expr(new_env, value_ast)

    return _eval_expr(new_env, body)

def _eval_atom(env, atom):
    if atom.ofType(SymbolAST):
        return LispSymbol(atom)
    elif atom.ofType(QuotedListAST):
        xs = atom.children[0]

        # again a function call with no args
        # is empty list
        # this can be parsed as is, with consistency with the 'scm' interpreter
        if xs == []:
            return LispEmptyList()

        ys = []
        for x in xs:
            if type(x) in _literals:
                ys.append(_eval_expr(env, x))
            else:
                ys.append(x)
        return f_list(*ys)
    elif atom.ofType(StringAST):
        return LispString(atom.children[0])
    elif atom.ofType(NumAST):
        real,imag = atom.children
        if imag == 0:
            return LispNum(real)
        else:
            # use python's built-in complex numbers
            return LispNum(real + imag*1j)
    elif atom.ofType(NameAST):
        name, = atom.children
        if name not in env:
            raise ValueError('EvalError: Name `{}` undefined.'.format(name))
        return env[name]
    else:
        b, = atom.children
        return LispBool(b)

### Environment Functions
def f_cons(l, r): return LispPair(l, r)
def f_car(pair): return pair._lhs
def f_cdr(pair): return pair._rhs
def f_list(*args):
    if len(args) != 0:
        l = LispEmptyList()
        for x in reversed(args):
            l = LispPair(x, l)
        return l
    else:
        return LispEmptyList()
    
# pointer equality by checking whether the two LispValue's 
# are actually the same PyObject
def f_p_eq(a, b):
    if isinstance(a, LispSymbol):
        return LispBool(a == b)
    return LispBool(a is b)

def f_eq(a, b): return LispBool(a == b)
def f_eq_num(a, b):
    if (type(a), type(b)) != (LispNum, LispNum):
        raise ValueError('EvalError: (=) expects numbers')
    return LispBool(a == b)
def f_sub(*x):
    if len(x) == 1:
        return LispNum(-x[0].value)
    return LispNum(reduce(lambda a, b: a - b, map(get_value,x)))

def compose(*fs):
    '''Compose functions together
    returning a single function.
    i.e. ``compose(f,g,h)(x) == f(g(h(x)))``
    '''
    def _wrapper(*args,**kwargs):
        *gs, f = fs
        v = f(*args, **kwargs)
        for g in reversed(gs):
            v = g(v)
        return v
    return _wrapper

# abstractproperties cannot be mapped over
def get_value(x):
    return x.value

def is_list(x):
    if type(x) is LispEmptyList:
        return True
    elif type(x) is LispPair:
        l,r = x.children
        return is_list(r)
    return False

def list_to_symbol(x):
    '''Converts a list of :class:`SymbolAST`, :class:`LispSymbol` or lists of symbols 
    to a SymbolAST which can be evaluated by :func:`evaluate`
    '''
    if type(x) == SymbolAST:
        return x.children[0]
    elif type(x) == LispSymbol:
        # extract AST and try again
        return list_to_symbol(x.value)
    elif type(x) == QuotedListAST:
        return dequoteify(x)
    elif type(x) == LispPair:
        # here we have a list of symbols
        xs = x.value_list
        return FuncApplicationAST(*map(list_to_symbol, xs))
    elif type(x) == LispString:
        return StringAST(x.value)
    elif type(x) == LispNum:
        return NumAST(x.value.real, x.value.imag)
    elif type(x) == LispBool:
        return BoolAST(x.value)

def evaluate(ast, defaultEnv=None):
    '''Evaluation on some given :class:`AST` *ast*
    Returning the python object which best represents the output
    
    If the interpreter for any reason cannot determine semantics, this raises `~ValueError`
    '''
    if not defaultEnv:
        defaultEnv = {
            '+': lambda *x: LispNum(sum(map(get_value, x))),
            '*': lambda *x: LispNum(reduce(lambda a,b: a*b, map(get_value, x))),
            '-': f_sub,
            '/': lambda *x: LispNum(reduce(lambda a,b: a/b, map(get_value, x))),
            'exp': lambda x: LispNum(exp(x.value)),
            'log': lambda x: LispNum(log(x.value)),

            # 'type' predicates
            'eq?': f_p_eq,
            'eqv?': f_eq,
            'equal?': f_eq,
            'list?': lambda x: LispBool(type(x) is LispPair and is_list(x)),
            'pair?': lambda x: LispBool(type(x) is LispPair),
            'integer?': lambda x: LispBool(type(x) is LispNum),
            'number?': lambda x: LispBool(type(x) is LispNum),
            'real?': lambda x: LispBool(type(x) is LispNum and x.children[1] is 0),
            'complex?': lambda x: LispBool(type(x) is LispNum),
            'symbol?': lambda x: LispBool(type(x) is LispSymbol),
            'string?': lambda x: LispBool(type(x) is LispString),
            'boolean?': lambda x: LispBool(type(x) is LispBool),

            '=': f_eq_num,
            'list': f_list,
            'cons': f_cons,
            'car': f_car,
            'cdr': f_cdr,
            'cddr': compose(f_cdr, f_cdr),
            'cdddr': compose(f_cdr, f_cdr, f_cdr),
            'cadr': compose(f_car, f_cdr),
            'caddr': compose(f_car, f_cdr, f_cdr),
            'cadddr': compose(f_car, f_cdr, f_cdr, f_cdr),

            'eval': lambda *x: defaultEnv, # need to put defaultEnv in the co_freevars field
        }

        # inject the environment into the call to eval
        # so we don't rebuild defaultEnv every time (eval ...) happens.
        defaultEnv['eval'] = partial(evaluate, defaultEnv=defaultEnv)

    # ast may not be an ast
    #  it may be a literal
    if type(ast) == LispPair:
        # convert to SymbolAST of FuncApplyAST 
        # and try evaluate that.
        return _eval_expr(defaultEnv, list_to_symbol(ast))

    return _eval_expr(defaultEnv, ast)

class Lisp(Plugin):
    '''Performs evaluations on given strings interpreted as a scheme/lisp input
    '''
    matches = build_lexchar_dict(**SYMBOL_LUT)

    def _build_program_dict(self):
        lexeme_dict = LexemeDict(Lisp.matches)
        program_dict = ProgramDict(SYMBOLS, lexeme_dict)
        return program_dict

    def _lex(self, s):
        program_dict = self._build_program_dict()
        return lex(program_dict, s)

    def _parse(self, tks):
        return parse(tks)

    def _eval(self, s):
        tks = self._lex(s)
        ast = self._parse(tks)
        return evaluate(ast)

    @Plugin.command('eval')
    def do_eval(self, e):
        '''Evaluate the data in *e* as if it were a lisp string
        Sending the reply
        '''
        try:
            r = self._eval(e['data'])
            e.reply(r.pretty_out())
        except ValueError as e:
            e.reply(str(e))
        except RuntimeError as e:
            e.reply('Error: Program halted at runtime.')


if __name__ == '__main__':
    ## REPL
    matches = build_lexchar_dict(**SYMBOL_LUT)
    symbols = SYMBOLS

    lexeme_dict = LexemeDict(matches)
    program_dict = ProgramDict(symbols, lexeme_dict)

    c = 0
    s = []
    while True:
        ss = input('> ')
        s.append(ss)
        c += len(list(filter(lambda x: x in ('(', '['), list(ss))))
        c -= len(list(filter(lambda x: x in (')', ']'), list(ss))))

        # only evalute matching brackets
        if not c:
            s = ' '.join(s)
            try:
                tks = lex(program_dict, s)
                tks2 = lex(program_dict, s)
                ast = parse(tks)
                res = evaluate(ast)
                print(res.pretty_out())
            except ValueError as e:
                print(str(e))
            except RuntimeError as e:
                print(str(e))
            finally:
                s = []
                c = 0
