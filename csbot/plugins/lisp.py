import string
import contextlib

from csbot.plugin import Plugin

SYMBOLS = '()[]'

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
    '''
    def __init__(self, value):
        self.value = value

    @property
    def type(self):
        return 'UNKNOWN_VALUE_TYPE'

class LispStr(Value):
    @property
    def type(self):
        return 'string'

    def __str__(self):
        return '"{}"'.format(str(self.value))

class LispNum(Value):
    @property
    def type(self):
        return 'num'

    def __str__(self):
        return str(self.value)

class LispName(Value):
    '''TODO: Remove this type
    for it only serves as a catch for missing environments at the moment
    '''
    @property
    def type(self):
        return 'TEMP_NAME'

    def __str__(self):
        return '<TEMP_NAME: {}>'.format(self.value)

class LispFunction(Value):
    def __init__(self, name, args):
        self.value = (name, args)

    @property
    def type(self):
        return 'lambda'

    def __str__(self):
        return '<lambda>'

class LispBool(Value):
    @property
    def type(self):
        return 'boolean'

    def __str__(self):
        if self.value:
            return '#t'
        return '#f'

def eval_function(call_pos, func):
    if not isinstance(func, LispFunction):
        raise InterpreterError(call_pos, 'Cannot call something of type <{}>'.format(func.type))

    name, args = func.value

    '''Eval a LispFunction'''
    # TODO: Use Environments and LispFunction types
    assert isinstance(name, LispName)
    if name.value == 'print':
        return '\t'.join(map(str, args))

    # TODO: Do this with Environments
    raise InterpreterError(call_pos, 'Unknown Function Name `{}`'.format(name))

class Parser:
    '''S-Expression Parser/Interpreter for a Lisp-like language
    '''
    def __init__(self):
        self.stream_position = 0
        self.tokens = []
        self.stream = None
        self._next_lexeme = None
        self._lexeme_gen = None
        self.auto_interpret = True

        self._ast = []
        self._parsing_str = False

    def push(self, x):
        '''Push a node 'x' onto the AST stack
        '''
        if self.auto_interpret:
            self._ast.append(x)

    def pop(self):
        '''Pops the last node off the AST stack and returns it
        '''
        if self.auto_interpret:
            return self._ast.pop()

    @contextlib.contextmanager
    def disable_interpreter(self):
        x = self.auto_interpret
        self.auto_interpret = False
        yield
        self.auto_interpret = x

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

        return self._parse()

    def _parse(self):
        if self.next == '(':
            self._parse_func_call()
        elif any(map(self.next.startswith, '0123456789')):
            self._parse_num()
        elif self.next.startswith('"'):
            self._parse_str()
        elif self.next.startswith('#'):
            self._parse_hash()
        else:
            self._parse_name()

    def _parse_hash(self):
        if not self.next.startswith('#'):
            raise ParseError(self.stream_position, "Expected to see a #-defined-symbol before here")  # this is a bad message.

        _, hash, s = self.next.partition('#')
        if s == 't':
            self.accept('#t')
            self.push(LispBool(True))
        elif s == 'f':
            self.accept('#f')
            self.push(LispBool(False))
        else:
            raise ParseError(self.stream_position, "Expected one of '#t', '#f'")

    def _parse_func_call(self):
        pos = self.stream_position

        self.accept('(')
        self._parse()
        name = self.pop()

        if isinstance(name, LispName):
            if name.value == 'if':
                return self._parse_if()

        args = []
        while self.next != ')':
            self._parse()
            args.append(self.pop())
        self.accept(')')
        self.push(eval_function(pos, LispFunction(name, args)))

    def _parse_if(self):
        '''Parses an (if c t e)
        starting from    ^
        '''
        pos = self.stream_position

        self._parse()
        condition = self.pop()
        if not isinstance(condition, LispBool):
            raise InterpreterError(pos, 'Condition must be boolean type, not `{}` type'.format(condition.type))

        if condition.value:
            self._parse()

            with self.disable_interpreter():
                self._parse()
        else:
            with self.disable_interpreter():
                self._parse()

            self._parse()

        self.accept(')')


    def _parse_num(self):
        num = self.next
        try:
            n = int(num)
            self.accept(num)
            self.push(LispNum(n))
        except ValueError:
            raise ParseError(self.stream_position, 'Expected an integer: [0-9]*')

    def _parse_str(self):
        s = self.next
        if s.startswith('"'):
            self.accept(s)
            self.push(LispStr(s[1:-1]))
        else:
            raise ParseError(self.stream_position, 'Expected string')

    def _parse_name(self):
        name = self.next
        self.accept(name)
        self.push(LispName(name))



class Lisp(Plugin):
    def lisp_eval(self, s, offset=0):
        parser = Parser()

        try:
            parser.parse(s)
            return str(parser.pop())
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
