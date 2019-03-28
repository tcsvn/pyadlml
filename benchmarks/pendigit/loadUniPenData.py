import re

def loadUnipenData(filename):
    fid = open(filename,'r')
    data = []
    labels = []
    noskip = True
    while True:
        if noskip:
            line = fid.readline()
        else:
            noskip = True
        if not line: break
        line = line.strip()
        if re.match('.SEGMENT.*\?.*"(\d)"', line):
            m = re.search('"(\d)"', line)
            num = m.group().replace('"', '')
            line = fid.readline()
            trace = []
            while not re.match('.SEGMENT.*\?.*"(\d)"', line):
                line = line.strip()
                if (re.match('(?P<x>[0-9]*)  (?P<y>[0-9]*)', line)):
                    m = re.match('(?P<x>[0-9]*)  (?P<y>[0-9]*)', line)
                    split = line.split(' ')
                    x = split[0]
                    y = split[-1]
                    trace.append([float(x), float(y)])
                elif (line == '.PEN_DOWN'):
                        trace.append([-1., 1.])
                elif (line == '.PEN_UP'):
                    trace.append([-1., -1.])
                line = fid.readline()
                if (re.match('.SEGMENT.*\?.*"(\d)"', line)):
                    noskip = False
                if not line: break
            data.append(trace)
            labels.append(int(num))
    fid.close()
    return data, labels