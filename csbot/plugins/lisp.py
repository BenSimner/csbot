import string
import contextlib
import collections

from csbot.plugin import Plugin

OPEN_BRACKET_SYMBOLS = '(['
CLOSED_BRACKET_SYMBOLS = ')]'
BRACKET_SYMBOLS = OPEN_BRACKET_SYMBOLS + CLOSED_BRACKET_SYMBOLS
SYMBOLS = '' + BRACKET_SYMBOLS

IO = []

class TokenizeError(Exception):
    def __init__(self, position, s):
        self.position = position
        super().__init__(s)

class ParseError(Exception):
    def __init__(self, position, s):
        self.position = position
        super().__init__(s)

class InterpreterError(Exception):
    def __init__(self, position, s):
        self.position = position
        super().__init__(s)

class Value:
    '''A Lisp value

    A Wrapper with a type and a value

    Handles pretty-printed output to terminal.
    '''
    type = 'UNKNOWN_VALUE_TYPE'

    def __init__(self, value):
        self.value = value

class LispStr(Value):
    type = 'string'

    def __str__(self):
        return '"{}"'.format(str(self.value))

class LispNum(Value):
    type = 'num'

    def __str__(self):
        return str(self.value)

class LispName(Value):
    '''TODO: Remove this type
    for it only serves as a catch for missing environments at the moment
    '''
    type = 'TEMP_NAME'

    def __str__(self):
        return '<TEMP_NAME: {}>'.format(self.value)

class LispFunction(Value):
    type = 'function'

    def __init__(self, args, block):
        self.value = (args, block)

    def __str__(self):
        return '<function>'

class LispLambda(LispFunction):
    def __str__(self):
        return '<lambda>'

class LispProcedure(LispFunction):
    type = 'procedure'

    def __init__(self, name, args, block):
        self.name = name
        super().__init__(args, block)

    def __str__(self):
        return '<procedure: {}>'.format(self.name)

class LispBool(Value):
    type = 'boolean'

    def __str__(self):
        if self.value:
            return '#t'
        return '#f'

class LispCons(Value):
    type = 'pair'

    def __init__(self, lhs, rhs):
        self.value = (lhs, rhs)

    def __str__(self):
        lhs, rhs = self.value
        lst = []

        # if lhs is None, then empty-list
        if lhs:
            lst.append(str(lhs))

        while rhs:
            if isinstance(rhs, LispCons):
                lhs, rhs = rhs.value

                if lhs:
                    lst.append(str(lhs))
            else:
                # end of pair
                lst.append('.')
                lst.append(str(rhs))
                break

        return "'({})".format(' '.join(lst))

def _assert_type(x, t):
    if not isinstance(x, t):
        raise InterpreterError('Expected an instance of `{}` (but got a `{}`)'.format(t.type, x.type))

def _sum(args):
    x = 0
    for v in args:
        x += v.value
    return LispNum(x)

def _neg(args):
    it = iter(args)
    x = next(it).value
    for v in it:
        x -= v.value
    return LispNum(x)

def _eq(args):
    it = iter(args)
    x = next(it).value
    b = True
    for v in it:
        b = b and (x == v.value)
    return LispBool(b)

def _print(args):
    arg, = args
    IO.append(str(arg))

def _cons(args):
    x, ys = args
    return LispCons(x, ys)

def _car(args):
    cons, = args
    _assert_type(cons, LispCons)
    return cons.lhs

def _cdr(args):
    cons, = args
    _assert_type(cons, LispCons)
    return cons.rhs

def _list(args):
    cc = LispCons(None, None)  # start with empty list
    for x in reversed(args):
        cc = LispCons(x, cc)

    return cc

class Parser:
    '''S-Expression Parser/Interpreter for a Lisp-like language
    '''
    def __init__(self):
        self.stream_position = 0
        self.tokens = []
        self.global_environment = {
            'print': LispProcedure('print', ['arg'], _print),
            'cons': LispProcedure('cons', ['x', 'ys'], _cons),
            'car': LispProcedure('car', ['xs'], _car),
            'cdr': LispProcedure('cdr', ['xs'], _cdr),
            'list': LispProcedure('list', ..., _list),
            '+': LispProcedure('+', ..., _sum),
            '-': LispProcedure('-', ..., _neg),
            '=': LispProcedure('=', ..., _eq),
        }

        self.stream = None
        self._next_lexeme = None
        self._lexeme_gen = None

        # stack of environment scopes
        self._scopes = [self.global_environment]

        # helper for tokenizer for splitting up a string
        self._parsing_str = False

        # the AST is a stack of functions
        self._ast = []

    def push(self, x):
        '''Push a node 'x' onto the AST stack
        '''
        self._ast.append(x)

    def pop(self):
        '''Pops the last node off the AST stack and returns it
        '''
        return self._ast.pop()

    def current_char(self):
        '''Gets the current char from the stream'''
        try:
            return self.stream[self.stream_position]
        except IndexError:
            return None

    def lexemes(self):
        '''Yields each token in sequence'''

        while True:
            try:
                x = self.stream[self.stream_position]
            except IndexError:
                raise TokenizeError(self.stream_position, 'Unexpected EOF')

            if x in SYMBOLS:
                yield from self._tokenize_symbol(x)
            elif x == '"':
                yield from self._tokenize_string()
            elif x == '#':
                yield from self._tokenize_hash()
            elif x in '0123456789':
                yield from self._tokenize_num(x)
            elif x in string.whitespace:
                self.stream_position += 1
            else:
                yield from self._tokenize_name()

    def _tokenize_name(self):
        x = self.current_char()
        lexeme = ''
        while x and not x in string.whitespace:
            if x in SYMBOLS:
                yield lexeme
                return

            lexeme += x
            self.stream_position += 1
            x = self.current_char()

        self.stream_position += 1
        yield lexeme

    def _tokenize_symbol(self, x):
        self.stream_position += 1
        yield x

    def _tokenize_string(self):
        self.stream_position += 1  # skip the #
        x = self.current_char()

        s = ''
        while x and x != '"':
            s += x
            self.stream_position += 1
            x = self.current_char()

        self.stream_position += 1
        yield '"{}"'.format(s)

    def _tokenize_hash(self):
        self.stream_position += 1
        x = self.current_char()

        if x == 't':
            yield '#t'
            self.stream_position += 1
        elif x == 'f':
            yield '#f'
            self.stream_position += 1
        else:
            raise TokenizeError(self.stream_position, 'Expected #t or #f with # symbol')

    def _tokenize_num(self, x):
        n = ''

        while x and x in '0123456789':
            n += x

            self.stream_position += 1
            x = self.current_char()
        yield n

    @property
    def next(self):
        if not self._next_lexeme:
            self._next_lexeme = next(self._lexeme_gen)
            self.tokens.append(self._next_lexeme)
        return self._next_lexeme

    def accept(self, x):
        '''
        Either accept 'x' or raise a ParseError
        :param x: the string to accept next
        :return:
        '''
        _next = self.next
        if _next == x:
            self._next_lexeme = None
        else:
            raise ParseError(self.stream_position, 'Expected {}'.format(x))

    def accept_one(self, xs):
        '''
        Either accept one string from xs
         or fail with ParseError
        :param x:
        :return:
        '''
        for x in xs:
            try:
                self.accept(x)
                break
            except ParseError:
                continue
        else:
            raise ParseError(self.stream_position, 'Expected one of {}'.format(', '.join(xs)))

    def parse(self, s):
        self.stream_position = 0
        self.stream = s
        self._lexeme_gen = self.lexemes()

        while True:
            try:
                self.next
            except TokenizeError:
                break

            self._parse()

    @contextlib.contextmanager
    def in_brackets(self):
        '''
        Accepts matching brackets
        '''
        if self.next == '(':
            self.accept('(')
            yield
            self.accept(')')
        else:
            self.accept('[')
            yield
            self.accept(']')

    def _parse(self):
        if self.next in OPEN_BRACKET_SYMBOLS:
            self._parse_func_call()
        elif any(map(self.next.startswith, '0123456789')):
            self._parse_num()
        elif self.next.startswith('"'):
            self._parse_str()
        elif self.next.startswith('#'):
            self._parse_hash()
        else:
            self._parse_var()

    def _parse_hash(self):
        if not self.next.startswith('#'):
            raise ParseError(self.stream_position, "Expected to see a #-defined-symbol before here")  # this is a bad message.
        _, _, s = self.next.partition('#')
        if s == 't':
            self.accept('#t')
            self.push(lambda: LispBool(True))
        elif s == 'f':
            self.accept('#f')
            self.push(lambda: LispBool(False))
        else:
            raise ParseError(self.stream_position, "Expected one of '#t', '#f'")

    def _print_stream(self, fmt, *args):
        s = ''

        for i, x in enumerate(self.stream):
            if i == self.stream_position:
                s += '⟦{}'.format(x)
            else:
                s += str(x)

        print('[{}] {}'.format(fmt.format(*args), s))

    def _parse_func_call(self):
        pos = self.stream_position
        with self.in_brackets():
            # special-case lazy-evaluated variants
            if self.next == 'if':
                self.accept('if')
                self._parse_if()
            elif self.next == 'lambda':
                self.accept('lambda')
                self._parse_lambda()
            elif self.next == 'define':
                self.accept('define')
                self._parse_define()
            else:
                self._parse()

                func_f = self.pop()

                argfs = []
                while self.next not in CLOSED_BRACKET_SYMBOLS:
                    self._parse()
                    argfs.append(self.pop())

                def _eval():
                    func = func_f()
                    if not isinstance(func, LispFunction):
                        raise InterpreterError(pos, 'Can only call Functional types (procedure/lambda) not `{}` types'.format(func.type))

                    (params, body) = func.value

                    args = []
                    for argf in argfs:
                        args.append(argf())
                    if params is not ... and len(params) != len(args):
                        raise InterpreterError(pos, 'Parameter mismatch, expected {} arguments (got {})'.format(len(params), len(args)))
                    return body(args)

                self.push(_eval)

    def _parse_if(self):
        '''Parses an (if c t e)
        starting from    ^
        '''
        pos = self.stream_position

        self._parse()
        condition_f = self.pop()

        self._parse()
        self._parse()

        else_block_f = self.pop()
        then_block_f = self.pop()

        def _eval():
            condition = condition_f()
            if not isinstance(condition, LispBool):
                raise InterpreterError(pos, 'Condition must be boolean type, not `{}` type'.format(condition.type))

            if condition.value:
                return then_block_f()
            else:
                return else_block_f()

        self.push(_eval)

    def _parse_lambda(self):
        '''Parses a (lambda (x y z) aexpr)
        starting from       ^
        '''
        argfs = []
        with self.in_brackets():
            while self.next not in CLOSED_BRACKET_SYMBOLS:
                self._parse_name()
                argfs.append(self.pop())
        self._parse()
        expr_f = self.pop()

        def _eval():
            names = []
            for x in argfs:
                name = x().value
                names.append(name)

            def _f_expr(args):
                with self.scope() as new_env:
                    for name, arg in zip(names, args):
                        new_env[name] = arg

                    return expr_f()

            return LispLambda(names, _f_expr)
        self.push(_eval)

    def _parse_define(self):
        '''Parses a (define (f x) ...)
        starting from       ^
        '''
        with self.in_brackets():
            self._parse_name()
            name_f = self.pop()

            argfs = []
            if self.next is '.':
                argfs = ...
            else:
                while self.next not in CLOSED_BRACKET_SYMBOLS:
                    self._parse_name()
                    argfs.append(self.pop())

        self._parse()
        expr_f = self.pop()

        def _eval():
            proc_name = name_f().value
            params = ... if argfs is ... else [f().value for f in argfs]
            proc = LispProcedure(proc_name, None, None) ## temp

            def _func_f(args):
                # TODO: VARARGS BETTER?
                if params is ...:
                    return expr_f()

                with self.scope() as new_env:
                    for name, arg in zip(params, args):
                        new_env[name] = arg
                    new_env[proc_name] = proc
                    return expr_f()

            proc.value = (params, _func_f)
            self._scopes[-1][proc_name] = proc
            return proc  # ???

        self.push(_eval)

    def _parse_num(self):
        num = self.next
        try:
            n = int(num)
            self.accept(num)
            self.push(lambda: LispNum(n))
        except ValueError:
            raise ParseError(self.stream_position, 'Expected an integer: [0-9]*')

    def _parse_str(self):
        s = self.next
        if s.startswith('"'):
            self.accept(s)
            self.push(lambda: LispStr(s[1:-1]))
        else:
            raise ParseError(self.stream_position, 'Expected string')

    def _parse_name(self):
        name = self.next
        self.accept(name)
        self.push(lambda: LispName(name))

    def _parse_var(self):
        pos = self.stream_position
        self._parse_name()
        lname_f = self.pop()

        def _eval():
            lname = lname_f()
            name  = lname.value

            for sc in reversed(self._scopes):
                if name in sc:
                    return sc[name]

            raise InterpreterError(pos, 'Unknown Name `{}`'.format(name))
        self.push(_eval)

    @contextlib.contextmanager
    def scope(self):
        ns = {}
        self._scopes.append(ns)
        yield ns
        self._scopes.pop()

class Lisp(Plugin):
    def lisp_eval(self, s, offset=0):
        IO.clear()
        parser = Parser()

        try:
            parser.parse(s)
            outfs = collections.deque()
            while True:
                try:
                    f = parser.pop()
                    outfs.appendleft(f)
                except IndexError:
                    break

            outs = []
            for f in outfs:
                outs.append(f())

            if outs[-1]:
                IO.append(str(outs[-1]))

            return '\t'.join(IO)
        #except Exception: raise
        except TokenizeError as e:
            return ' ' * (offset + e.position) + '^ TokenizeError: ' + str(e)
        except ParseError as e:
            return ' ' * (offset + e.position) + '^ ParseError: ' + str(e)
        except InterpreterError as e:
            return ' ' * (offset + e.position) + '^ InterpreterError: ' + str(e)
        except Exception as e:
            self.log.debug('[lisp_eval] Unknown Exception: {}'.format(e))
            return 'Unknown Exception (see logs)'

    @Plugin.command('eval', help='For interpreting, not calculating')
    def eval(self, e):
        '''Do you not have a lisp?
        '''
        msg = e['message']
        data = e['data']
        offset = msg.index(data)
        e.reply(self.lisp_eval(data, offset=offset))