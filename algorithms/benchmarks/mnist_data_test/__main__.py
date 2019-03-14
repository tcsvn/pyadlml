import argparse

parser = argparse.ArgumentParser(
    prog='pendigits-hmm',
    description='python pen-based handwritten digits recognition (pendigits ' \
                'data set) using a hidden markov model (HMM)'
)

parser.add_argument('--path', default='data',
                    help='path to the pendigits data')

args = parser.parse_args()


def main():

    print('Hello world')

if __name__ == '__main__':

    main()
