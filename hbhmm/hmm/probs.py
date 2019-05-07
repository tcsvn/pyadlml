
# as the model is unstable create an
# own data type to emulate numeric type
# of a probability
import operator as op
from typing import List

import numpy as np
import sys
from scipy.special import logsumexp
LOGZERO = -sys.float_info.max


class Probs(object):


    def __init__(self, prob):
        try:
            prob = float(prob)
        except:
            raise ValueError
        # todo
        # this is suspended because of summing probabilities
        # greater than one and then dividing
        #if prob > 1.0:
        #    raise ValueError
        else:
            self.prob = self.eln(prob)

    @classmethod
    def np_prob_arr2R(cls, np_arr):
        return np.exp(np_arr).astype(np.float, copy=False)


    def exp(self):
        """
        fct is called by np.exp(x) from numpy
        :return:
        """
        return self.eexp(self.prob)

    @ classmethod
    def eexp(cls, x : float):
        """
        required for np.exp(arr)
        :return:
        """
        if x == LOGZERO:
            return 0
        else:
            return np.exp(x)

    @ classmethod
    def eln(self, x : float):
        if x == 0.0:
            return LOGZERO
        elif x > 0.0:
            return np.log(x)
        else:
            raise ValueError

    def __str__(self):
        s = ""
        s += str(self.prob)
        return s

    def prob_to_norm(self):
        return self.eexp(self.prob)

    """
    implement arithmetic operations
    
    """
    #def __add__(self, other):
    #    """
    #    x + y
    #    :param other:
    #    #log(a + b) = log(a * (1 + b / a)) = log(a) + log(1 + b / a)
    #    :return: new Probs object
    #    """
    #    # choose least negative operater to be a
    #    if self.prob > other.prob:
    #        a = self.prob
    #        b = other.prob
    #    else:
    #        a = other.prob
    #        b = self.prob

    #    res = Probs(1.0)
    #    res.prob = a + np.log(1 + np.exp(b)/np.exp(a))
    #    return res

    def __add__(self, other):
        """
        x + y
        :param other:
        #log(a + b) = log(a * (1 + b / a)) = log(a) + log(1 + b / a)
        :return: new Probs object
        """
        res = Probs(1.0)
        if other.prob == LOGZERO or self.prob == LOGZERO:
            if self.prob == LOGZERO:
                res.prob = other.prob
            else:
                res.prob = self.prob
        else:
            res.prob = logsumexp([self.prob, other.prob])
            # todo why does above work at all???
            # stackoverflow:  https://stackoverflow.com/questions/778047/we-know-log-add-but-how-to-do-log-subtract
            #if self.prob > other.prob:
            #    a = self.prob
            #    b = other.prob
            #else:
            #    a = other.prob
            #    b = self.prob
            #res.prob = a + np.log1p(np.exp(b-a))
        return res


    def __sub__(self, other):
        """
        x - y
        :param other:
        #log(a - b) = log(a * (1 - b / a)) = log(a) + log(1 - b / a)
        :return: new Probs object
        """
        # choose least negative operater to be a
        #if self.prob > other.prob:
        #    a = self.prob
        #    b = other.prob
        #else:
        #    a = other.prob
        #    b = self.prob

        # res = Probs(1.0)
        # res.prob = a + np.log(1 - np.exp(b)/np.exp(a))
        # res = Probs(1.0)
        # res.prob = logsumexp(self.prob, other.prob)
        # res.prob = np.logaddexp(self.prob,other.prob) # numpy's version of log add
        # res.prob = np.logaddexp(self.prob,-other.prob) # numpy's version of log add
        # res.prob = np.logaddexp() # numpy's version of log add
        # return res
        # todo verify
        # stackoverflow:  https://stackoverflow.com/questions/778047/we-know-log-add-but-how-to-do-log-subtract
        res = Probs(1.0)
        #res.prob = self._logsubexp(self.prob, other.prob)
        a = self.prob
        #print(a)
        b = other.prob
        #print(b)
        #b = -b
        #print(b)
        #res.prob = logsumexp(a,b)
        res.prob = self._logsubexp(a,b)
        return res

    def _logsubexp(self, a, b):
        #return self._logsubexp_factor_method(a,b)
        #return self._logsubexp_factor_method2(a,b)
        return self._logsubexp_stackoverflow_method(a,b)

    def _logsubexp_factor_method(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        z = min(a,b)
        a = a - z
        b = b - z
        if a < b:
            raise ValueError
        #print('--'*20)
        #print(a,b)
        #print(np.exp(a) - np.exp(b))
        res = np.log(np.exp(a) - np.exp(b))
        #print(res)
        return res + z



    def _logsubexp_stackoverflow_method(self, a, b):
        """
            https://stats.stackexchange.com/questions/383523/subtracting-very-small-probabilities-how-to-compute

        :param a: self.prob
        :param b: other.prob
        :return:
        """
        if a == LOGZERO or b == LOGZERO:
            if a == LOGZERO:
                return b
            else:
                return a
        else:
            # swap the variables in order that
            if a <= b:
                c = a
                a = b
                b = c
            suma1 = a
            suma2 = np.log1p(-np.exp(b-a))
            if suma2 == np.NaN or suma2 == np.NAN:
                print('lalalala'*100)
            res = suma1 + suma2
            return res

    def _logsubexp_wiki_method(self, a, b):
        """
        method log-probabilities in
        :param a:
        :param b:
        :return:
        """
        return a + np.log1p(-np.exp(b-a))

    #def _logsubexp(self, X : [float]) -> float:
    #    # todo this is very dangerous change it !!!
    #    #if a < b:
    #    #    raise ValueError
    #    if sum(X) < 0:
    #        raise ValueError
    #    X = np.array(X)
    #    c = max(X)
    #    return np.log(np.sum(np.exp(X-c))) + c


    def __mul__(self, other):
        """
        x * y
        :param other:
        :return:
        """
        res = Probs(1.0)
        if self.prob == LOGZERO:
            res.prob = LOGZERO
            #raise ValueError
        else:
            if type(other) in [float, int]:
                res.prob = self.prob + self.eln(other)
            else:
                res.prob = self.prob + other.prob
        return res

    def __truediv__(self, other):
        """
        x * y
        :param other:
        :return:
        """
        res = Probs(1.0)
        res.prob = self.prob - other.prob
        return res

    def __pow__(self, power, modulo=None):
        """
        x ** y
        :param other:
        :return:
        """
        res = Probs(1.0)
        res.prob = power*self.prob
        return res

    def __matmul__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __divmod__(self, other):
        raise NotImplementedError

    """
    implement inplace arithmetic operations
    
    """
    def __iadd__(self, other):
        """
        x += y
        :param other:
        :return:
        """
        if type(other) in [float, int]:
            self.prob = logsumexp([self.prob, np.log(other)])
        else:
            self.prob = logsumexp([self.prob, other.prob])
        # obligatory return statement
        return self

    def __isub__(self, other):
        """
        x -= y
        :param other:
        :return:
        """
        self.prob = self._logsubexp(self.prob, other.prob)
        return self

    def __imul__(self, other):
        """
        x *= y
        :param other:
        :return:
        """
        # todo sometimes here is an overflow
        if self.prob == LOGZERO or other.prob == LOGZERO:
            self.prob = LOGZERO
        else:
            self.prob = self.prob + other.prob
        return self

    def __idiv__(self, other):
        """
        x /= y
        :param other:
        :return:
        """
        self.prob = self.prob - other
        return self

    def __ipow__(self, power, modulo=None):
        """
        x **= y     | x = x^y
        :param power:
        :return:
        """
        self.prob = power * self.prob
        return self

    # todo add all arithmetic but inplace
    """
    implement unary artihmetic operations
    """
    def __neq__(self):
        """
        raises Error
        :return:
        """
        #self.prob = -self.prob
        # a = -c <=> ln(a) = ln(-c) = ln(-1) + ln(c) Error
        print('this action is not allowed ')
        raise ValueError

    def __pos__(self):
        """
        + x
        :return:
        """
        return self

    def __abs__(self):
        """
        | x |
        :return:
        """
        return abs(self.prob)
        #print('this action is not allowed')
        #raise ValueError

    """
    implement built-in functions
    """
    def __int__(self):
        """
        emulates int(x)
        :return:
        """
        return int(self.prob)

    def __float__(self):
        """
        emulates float(x)
        :return:
        """
        return float(self.prob)

    def __round__(self, n=None):
        """
        emulates round(x)
        :param n:
        :return:
        """
        return round(self.prob, n)

    """
    comparision operators below
    the order is in log space is equal to real space
    """
    def comp_helper(self, fct, other):
        if type(other) in [float, int]:
            return fct(self.prob, other)
        else:
            return fct(self.prob, other.prob)

    def __lt__(self, other):
        """
        x < y
        :param other:
        :return:
        """
        return self.comp_helper(op.lt, other)

    def __le__(self, other):
        """
        x <=y
        :param other:
        :return:
        """
        return self.comp_helper(op.le, other)

    def __eq__(self, other):
        """
        x == y
        :param other:
        :return:
        """
        return self.comp_helper(op.eq, other)

    def __ne__(self, other):
        """
        x != y
        :param other:
        :return:
        """
        return self.comp_helper(op.ne, other)

    def __gt__(self, other):
        """
        x > y
        :param other:
        :return:
        """
        return self.comp_helper(op.gt, other)

    def __ge__(self, other):
        """
        x >= y
        :param other:
        :return:
        """
        return self.comp_helper(op.ge, other)
