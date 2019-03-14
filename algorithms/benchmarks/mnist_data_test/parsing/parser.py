from algorithms.benchmarks.mnist_data.parsing import digit

class Parser:

    def __init__(self):
        pass

    def parse_file(self, filename):

        digits = []

        reading_digit = False
        buff = []
        with open(filename, 'r') as f:
            for line in f:

                line = line.rstrip()

                if line.startswith('.SEGMENT'):
                    reading_digit = True
                if line == '':
                    reading_digit = False

                if reading_digit:
                    buff.append(line)
                elif len(buff) > 0:
                    dig = self.parse_digit(buff)
                    dig.normalise()
                    digits.append(dig)
                    buff = []

        return digits

    def parse_digit(self, lines):

        if len(lines) > 0 :
            dig = digit.Digit()

            started = False
            current_curve = []
            for line in lines:
                if line.startswith(' '):
                    points = line.split(' ')
                    first_point = next(s for s in points if s != '')
                    points.reverse()
                    last_point = next(s for s in points if s != '')
                    current_curve.append([int(first_point), int(last_point)])
                elif line.startswith('.PEN_UP'):
                    dig.add_curve(current_curve)
                    current_curve = []
                elif line.startswith('.SEGMENT'):
                    split = line.split(' ')
                    dig.set_label(split[-1][1:-1])
            return dig

        else:
            return []
