
# as the model is unstable create an
# own data type to emulate numeric type
# of a probability
import numpy as np
import sys
from scipy.special import logsumexp

class Probs(object):

    def __init__(self, prob):
        try:
            prob = float(prob)
        except:
            raise ValueError
        if prob < 0.0 or 1.0 < prob:
            raise ValueError
        if prob == 0.0:
            self.prob = - sys.float_info.max
        else:
            self.prob = np.log(prob) # normal case


    def prob_to_norm(self):
        if self.prob == - sys.float_info.max:
            print('went here')
            return np.nextafter(0, 1)
        else:
            return np.exp(self.prob)

    def __str__(self):
        s = ""
        s += str(self.prob)
        return s

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
        # choose least negative operater to be a
        #a = max(self.prob, other.prob)
        #b = min(self.prob, other.prob)

        res = Probs(1.0)
        res.prob = logsumexp([self.prob, other.prob])
        #res.prob = np.logaddexp(a,b) # numpy's version of log add
        return res

    # stackoverflow:  https://stackoverflow.com/questions/778047/we-know-log-add-but-how-to-do-log-subtract
#      double log_add(double x, double y) {
#    if(x == neginf)
#      return y;
#    if(y == neginf)
#      return x;
#    return max(x, y) + log1p(exp( -fabs(x - y) ));
#  }



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

        #res = Probs(1.0)
        #res.prob = a + np.log(1 - np.exp(b)/np.exp(a))
        #res = Probs(1.0)
        #res.prob = logsumexp(self.prob, other.prob)
        #res.prob = np.logaddexp(self.prob,other.prob) # numpy's version of log add
        #res.prob = np.logaddexp(self.prob,-other.prob) # numpy's version of log add
        #res.prob = np.logaddexp() # numpy's version of log add
        #return res
        # todo verify
        # stackoverflow:  https://stackoverflow.com/questions/778047/we-know-log-add-but-how-to-do-log-subtract
        res = Probs(1.0)
        res.prob = self._logsubexp(self.prob, other.prob)
        return res

    def _logsubexp(self, a, b):
        if a < b:
            raise ValueError
        if np.isneginf(b):
            return a
        else:
            return a + np.log1p(-np.exp(b - a))

    def __mul__(self, other):
        """
        x * y
        :param other:
        :return:
        """
        res = Probs(1.0)
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

    def __ipow__(self,power, modulo=None):
        """
        x **= y     | x = x^y
        :param other:
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
        - x
        :return:
        """
        # todo look up if ok !!!
        self.prob = -self.prob
        return self

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

    def __lt__(self, other):
        """
        x < y
        :param other:
        :return:
        """
        return self.prob < other.prob

    def __le__(self, other):
        """
        x <=y
        :param other:
        :return:
        """
        return self.prob <= other.prob

    def __eq__(self, other):
        """
        x == y
        :param other:
        :return:
        """
        return self.prob == other.prob

    def __ne__(self, other):
        """
        x != y
        :param other:
        :return:
        """
        return self.prob != other.prob

    def __gt__(self, other):
        """
        x > y
        :param other:
        :return:
        """
        return self.prob > other.prob

    def __ge__(self, other):
        """
        x >= y
        :param other:
        :return:
        """
        return self.prob >= other.prob
