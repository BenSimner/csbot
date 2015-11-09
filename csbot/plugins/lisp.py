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

# literals are atomic values
#  i.e. values that evaluate to themselves
LITERALS = (TokenType.STRING, TokenType.NUMBER, TokenType.REAL, 
            TokenType.IMAGINARY, TokenType.BOOL_TRUE, TokenType.BOOL_FALSE)

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

NameAST = make_ast('NameAST', ['name'])
BoolAST = make_ast('BoolAST', ['bool'])
NameAST.__repr__ = lambda self: str(self.children[0])
SymbolAST = make_ast('SymbolAST', ['expr'])
StringAST = make_ast('StringAST', ['string'])
FuncApplicationAST = make_ast('FuncApplyAST', ['funcName', '*args'])


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
        try:
            _CURRENT_TOKEN = next(TOKENS)
        except StopIteration:
            raise ValueError('ParseError: Unexepctedly reached end of stream')


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
        # TODO:
        # quoted literals are literals
        # quoted list is list of quotes
        # eval on above should coerce into ASTs
        accept(TokenType.APOSTROPHE)
        _parse_expr()
        ast = pop()
        push(SymbolAST(ast))
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

class LispValue(ABC):
    @abstractmethod
    def __repr__(self):
        pass

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

class LispNum(LispValue):
    def __init__(self, num):
        self._num = num

    @property
    def value(self):
        return self._num

    @property
    def children(self):
        return ()

    def __repr__(self):
        return repr(self.value)

class LispString(LispValue):
    def __init__(self, s):
        self._s = s

    @property
    def value(self):
        return self._s

    @property
    def children(self):
        return ()

    def __repr__(self):
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

    def __repr__(self):
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

    def __repr__(self):
        return repr(self._ast)

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

    def __repr__(self):
        lhs, rhs = self.children
        sb = '(' + repr(lhs)

        if type(rhs) == LispPair:
            while True:
                lhs, rhs = rhs.children
                sb += ' '
                sb += repr(lhs)
                if type(rhs) == LispEmptyList:
                    sb += ')'
                    break
                elif type(rhs) != LispPair:
                    sb += ' . '
                    sb += repr(rhs)
                    sb += ')'
                    break
        elif type(rhs) == LispEmptyList:
            sb += ')'
        else:
            sb += ' . {})'.format(repr(rhs))

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

    def __repr__(self):
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
            return _eval_let(env, bindings, body)

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
    newEnv = dict(env) 
    for binding in bindings:
        if not binding.ofType(FuncApplicationAST):
            raise ValueError('EvalError: `let` must take arguments in format (let [(b0 x0) (b1 x1)] [body])')
            
        name, value_ast = binding.children
        name, = name.children
        newEnv[name] = _eval_expr(env, value_ast)

    return _eval_expr(newEnv, body)

def _eval_let(env, bindings, body):
    # create a copy
    for binding in bindings:
        if not binding.ofType(FuncApplicationAST):
            raise ValueError('EvalError: `let` must take arguments in format (let [(b0 0) (b1 x1)] [body])')
            
        name, value_ast = binding.children
        name, = name.children
        env[name] = _eval_expr(env, value_ast)

    return _eval_expr(env, body)

def _eval_atom(env, atom):
    if atom.ofType(SymbolAST):
        # handle '() for emptylist
        try:
            f, = atom.children
            if f.ofType(FuncApplicationAST):
                args, = f.children
                if tuple(args) == ():
                    return LispEmptyList()
        except:
            pass
        return LispSymbol(atom)
    elif atom.ofType(StringAST):
        return LispString(atom.children[0])
    elif atom.ofType(NumAST):
        real,imag = atom.children
        if imag == 0:
            return LispNum(real)
        else:
            # use python's built-in complex numbers
            return LispNum(real + imag*(1.0j))
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
def f_p_eq(a, b): return LispBool(a is b)
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

def evaluate(ast, defaultEnv=None):
    '''Does semantical evaluation on some given :class:`AST` *ast*
    Returning the python object which best represents the output
    
    If the interpreter for any reason cannot determine semantics, this raies `~ValueError`
    '''
    if not defaultEnv:
        defaultEnv = {
            '+': lambda *x: LispNum(sum(map(get_value, x))),
            '*': lambda *x: LispNum(reduce(lambda a,b: a*b, map(get_value, x))),
            '-': f_sub,
            '/': lambda *x: LispNum(reduce(lambda a,b: a/b, map(get_value, x))),

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
            'cadr': compose(f_car, f_cdr),
            'caddr': compose(f_car, f_cdr, f_cdr),
            'cadddr': compose(f_car, f_cdr, f_cdr, f_cdr),

            'eval': lambda *x: defaultEnv, # need to put defaultEnv in the co_freevars property
        }

        # inject the environment into the call to eval
        # so we don't rebuild defaultEnv every time (eval ...) happens.
        defaultEnv['eval'] = partial(evaluate, defaultEnv=defaultEnv)
    return _eval_expr(defaultEnv, ast)

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
            e.reply(repr(r))
        except ValueError as e:
            e.reply(str(e))

if __name__ == '__main__':
    ## REPL
    matches = build_lexchar_dict(**SYMBOL_LUT)
    symbols = SYMBOLS

    lexeme_dict = LexemeDict(matches)
    program_dict = ProgramDict(symbols, lexeme_dict)

    def _count_brackets(s):
        if not s:
            return 0
        c, *s = s
        s = ''.join(s)
        if c in ('(', '['):
            return 1 + _count_brackets(s)
        if c in (')', ']'):
            return _count_brackets(s) - 1
        return _count_brackets(s)

    s = ''
    unmatched = 0
    while True:
        s_ = input('> ')
        s += s_
        unmatched = unmatched + _count_brackets(s_)
        if not unmatched:
            tks = lex(program_dict, s)
            ast = parse(tks)
            res = evaluate(ast)
            print(res)
            s = ''; k = 0
