
class Foo():
    def __init__(self):
        self.a = 1
        self.b = -1

class Wrapper(object):

    def test_setter(self, val):
        self._test = val

    def test_getter(self):
        return self._test

    def __init__(self, wrapped):
        self.wrapped = wrapped

        # copy every attribute into the wrapper
        for attr in self.wrapped.__dict__:
            self._add_property(attr, wrapped)


        # create new child class for TrainAndEvalOnlyWrapper that inherits from the both
        # this is done because an isinstance than can detect both
        new_class_name = self.__class__.__name__ + self.wrapped.__class__.__name__
        child_class = type(new_class_name, (self.__class__, wrapped.__class__), {})
        self.__class__ = child_class

    def _add_property(self, attr_name, wrapped):
        value = getattr(wrapped, attr_name)
        # create local fget and fset functions
        fget = lambda self: getattr(self, '_' + attr_name)
        # set the attribute of the wrapped also
        fset = lambda self, value: (setattr(wrapped, attr_name, value), setattr(self, '_' + attr_name, value))

        # add property to self
        setattr(self.__class__, attr_name, property(fget, fset))
        # add corresponding local variable
        setattr(self, '_' + attr_name, value)


class P(object):
    def __init__(self):
        self._add_property('x', 0)

    def _add_property(self, name, value):
        # create local fget and fset functions
        fget = lambda self: (print('entered get lambda'), getattr(self, '_' + name))[-1]
        fset = lambda self, value: (print('entered set lambda'),setattr(self, '_' + name, value))

        # add property to self
        setattr(self.__class__, name, property(fget, fset))
        # add corresponding local variable
        setattr(self, '_' + name, value)

    def f(self):
        print('entered f1')
        return self._x

    def f2(self, val):
        print('entered f2')
        self._x = val

    #x = property(lambda self: io(
    #        print('entered get lambda'),
    #        self._x
    #    )[-1],
    #    lambda self, value: (
    #        print('entered set lambda'),
    #        setattr(self, '_x', value)
    #    )
    #)

#p = P()
#print(type(p))
#print(p.x)
#p.x = 2
#print(p.x)
#
#print('~'*10)

f = Foo()
print('fa: ', f.a)
print('fb: ', f.b)
#
w = Wrapper(f)
print('iif: ', isinstance(w, Foo))
print('iiw: ', isinstance(w, Wrapper))

print('wa: ', w.a)
print('wb: ', w.b)

w.a = 10

print('wa: ', w.a)
print('fa: ', f.a)
print(f.__dict__)
print(w.__dict__)
#w.test = 1
#print('pt: ', w.test)
##print(f.__dict__)
##print(w.__dict__)
#