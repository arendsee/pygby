#!/usr/bin/env python3

import pygby
import unittest
from tempfile import NamedTemporaryFile
import csv
import sys

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
        t1 = (('a', 'b', 'c', 'd', 'e'),
              (12, 'e', -1, 'di', 1.66),
              (1, 'w', 5, 'la', -2),
              (12, 'd', -1, 'di', 1.34),
              (1, 'w', 5, 'la', -5),
              (8, 'e', -1, 'di', 1.34),
              (6, 'w', 3, 'la', 0),
              (6, 'w', 4.45, 'la', 1))

        self.infile = self.vector2file(t1)
        outfile = self.getemptyfile()

        defargs = {'count': True,
                'header': True,
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
        sys.stdin = open(self.infile.name, 'r')
        args = ['--groupby', '2',
                '--min', '3', '5',
                '-d', '\t',
                '--header']
        out = pygby.write(args, True)
        # a      b      c       d      e
        #
        # 12     d      -1      di     1.34
        # 12     e      -1      di     1.66
        # 8      e      -1      di     1.34
        # 6      w      3       la     0
        # 6      w      4.45    la     1
        # 1      w      5       la     -2
        # 1      w      5       la     -5
        print(out)
        # self.assertEquals(


        self.infile.close()

if __name__ == '__main__':
    unittest.main()
