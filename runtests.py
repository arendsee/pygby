#!/usr/bin/env python3

import pygby
import unittest
from tempfile import NamedTemporaryFile
import csv
import sys

def lol2dict(lol):
    z = tuple(zip(*lol[1:]))
    d = {lol[0][i]:z[i] for i in range(len(z))}
    return(d)

class Testpygby(unittest.TestCase):
    def vector2file(self, x):
        f1 = NamedTemporaryFile()
        x = ['\t'.join([str(z) for z in y]) for y in x]
        x = '\n'.join([y for y in x])
        with open(f1.name, 'wb') as f:
            f.write(bytes(x, 'ascii'))
        return(f1)

    def file2vector(self, infile):
        reader = csv.reader(infile, delimiter='\t')
        return(reader.readlines())

    def getemptyfile(self):
        return(NamedTemporaryFile())

    def setUp(self):
        # Headered
        # t1 = (('a', 'b', 'c', 'd', 'e'),
        #       (12, 'e', -1, 'di', 1.66),
        #       (1, 'w', 5, 'la', -2),
        #       (12, 'd', -1, 'di', 1.34),
        #       (1, 'w', 5, 'la', -5),
        #       (8, 'e', -1, 'di', 1.54),
        #       (6, 'w', 3, 'la', 0),
        #       (6, 'w', 4.45, 'la', 1))

        t1 = (( 'a', 'b', 'c', 'd', 'e'),
              ( 0  , 'z', 1.1, 'w', 1.0),
              ( 1  , 'z', 6.2, 'w', 2.0),
              ( 2  , 'z', 1.3, 'w', 3.0),
              ( 1  , 'z', 3.0, 'x', 2.0),
              ( 1  , 'z', 1.7, 'x', 5.0),
              ( 9  , 'z', 2.2, 'x', 2.0),
              ( 1  , 'y', 2.6, 'w', 0.0),
              ( 3  , 'y', 1.1, 'w', 1.0),
              ( 2  , 'y', 4.9, 'w', 5.0),
              ( 1  , 'y', 2.0, 'x', 1.0),
              ( 4  , 'y', -.5, 'x', 6.0),
              ( 5  , 'y', 3.0, 'x', 2.0))

        self.infile = self.vector2file(t1)
        outfile = self.getemptyfile()

        defargs = {'count': True,
                'header': True,
                'silent_header': False,
                'indel': '\t',
                'outdel': '\t',
                'ids': (3,),
                'max': (0,2),
                'min': None,
                'sum': None,
                'median': (0,),
                'sd': None,
                'mean': None,
                'smin': ({'by':0, 'record':(1,)},),
                'smax': None,
                'seq': None,
                'adj': None,
                'allids': {0,1,2},
                'floats': {0,2}
                }
        self.funman = pygby.FunManager(defargs)

    def test_numeric_operations(self):
        empty = ()

        fm = pygby.FunMap.funmap
        args = ('max', 'smax', 'min', 'smin', 'sum', 'mean', 'median')

        # Test correct output for numeric maps
        x = (50.0,-100, 0, -1.0)
        self.assertEqual(fm['max'](x), 50)
        self.assertEqual(fm['smax'](x), 50)
        self.assertEqual(fm['min'](x), -100)
        self.assertEqual(fm['smin'](x), -100)
        self.assertEqual(fm['sum'](x), -51)
        self.assertAlmostEqual(fm['mean'](x), -12.75)
        self.assertAlmostEqual(fm['median'](x), -0.5)
        self.assertAlmostEqual(fm['sd'](x), 62.85101, places=5)

        # Test when input is length 1
        x = (5.0,)
        self.assertEqual(fm['max'](x), 5.0)
        self.assertEqual(fm['smax'](x), 5.0)
        self.assertEqual(fm['min'](x), 5.0)
        self.assertEqual(fm['smin'](x), 5.0)
        self.assertEqual(fm['sum'](x), 5.0)
        self.assertAlmostEqual(fm['mean'](x), 5.0)
        self.assertAlmostEqual(fm['median'](x), 5.0)
        self.assertEqual(fm['sd'](x), 'NA')

        # Test when input is non-iterable
        x = 5.0
        for arg in args:
            self.assertRaises(TypeError, fm[arg], x)

        # Test when input is None
        x = None
        for arg in args:
            self.assertRaises(TypeError, fm[arg], x)

        # Test for TypeErrors when string gets in
        x = ('a',20,4)
        for arg in args:
            self.assertRaises(TypeError, fm[arg], x)

        # Test for TypeErrors when string gets in
        # This WILL fail for the max and min functions, since they accept all
        # string inputs. I won't add a check since that would cost too much.
        x = ('a','b')
        for arg in ('sum', 'mean', 'median', 'sd'):
            self.assertRaises(TypeError, fm[arg], x)

    def test_write(self):
        # 'a'   'b'    'c'    'd'    'e'
        # ------------------------------
        # 7     'z'    1.1    'w'    1.0
        # 1     'z'    6.2    'w'    2.0
        # 2     'z'    1.3    'w'    3.0
        # 1     'z'    3.0    'x'    2.0
        # 1     'z'    1.7    'x'    5.0
        # 9     'z'    2.2    'x'    2.0
        # 1     'y'    2.6    'w'    0.0
        # 3     'y'    1.1    'w'    1.0
        # 2     'y'    4.9    'w'    5.0
        # 1     'y'    2.0    'x'    1.0
        # 4     'y'    -.5    'x'    6.0
        # 5     'y'    3.0    'x'    2.0

        # Assert most basic function
        sys.stdin = open(self.infile.name, 'r')
        args = ['-g', '2',
                '--min', '3',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        self.assertEqual(out, [['b', 'c.min'], ['y', -0.5], ['z', 1.1]])

        # Assert grouping of data columns
        sys.stdin = open(self.infile.name, 'r')
        args = ['-g', '2',
                '--min', '5', '1',
                '--min', '5', '3',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        self.assertEqual(out, [['b', 'a.min', 'c.min', 'e.min'],
                               ['y', 1.0, -0.5, 0.0],
                               ['z', 0.0, 1.1, 1.0]])

        # Assert grouping of ids
        sys.stdin = open(self.infile.name, 'r')
        args = ['-g', '2', '4',
                '--min', '5', '1',
                '--min', '5', '3',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        self.assertEqual(out, [['b', 'd', 'a.min', 'c.min', 'e.min'],
                               ['y', 'w', 1.0, 1.1, 0],
                               ['y', 'x', 1, -.5, 1],
                               ['z', 'w', 0, 1.1, 1.0],
                               ['z', 'x', 1.0, 1.7, 2.0]])

        # Assert selection
        sys.stdin = open(self.infile.name, 'r')
        args = ['-g', '2',
                '--smax', '1', '3', '5',
                '--smax', '3', '5',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        self.assertEqual(out, [['b', 'c.where.a.smax', 'e.where.a.smax', 'e.where.c.smax'],
                               ['y', 3.0, '2.0', '5.0'],
                               ['z', 2.2, '2.0', '2.0']])

        # Assert calculations
        sys.stdin = open(self.infile.name, 'r')
        args = ['-g', '2',
                '--count',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        self.assertEqual(out, [['b', 'count'],
                               ['y', 6],
                               ['z', 6]])

        self.infile.close()

if __name__ == '__main__':
    unittest.main()
