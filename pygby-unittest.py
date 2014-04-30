#!/usr/bin/env python3

import pygby
import unittest
from tempfile import TemporaryFile
import csv

class Testpygby(unittest.TestCase):
    def setUp(self):
        self.testdatarow = ((1,2,3),(5,5,5),(1,10,5))

        # Headered
        t1 = (('a', 'b', 'c', 'd', 'e'),
              (12, 'e', -1, 'di', 1.66),
              (1, 'w', 5, 'la', -2),
              (12, 'd', -1, 'di', 1.34),
              (1, 'w', 5, 'la', -5),
              (8, 'e', -1, 'di', 1.34),
              (6, 'w', 3, 'la', 0),
              (6, 'w', 4.45, 'la', 1))
        t1_file = TemporaryFile()
        t1_out = TemporaryFile()

        with open(t1_file.name, 'w') as f:
            w = csv.writer(f, delimiter='\t')
            for row in t1:
                w.writerow(row)

        defargs = {'count': True,
                'indel': '\t',
                'max': (0,2),
                'allids': {0,1,2},
                'min': None,
                'smin': ({'by':0, 'record':(1,)},),
                'sum': None,
                'smax': None,
                'median': (0,),
                'ids': {3},
                'floats': {0,2},
                'header': True,
                'outdel': '\t',
                'sd': None,
                'in': t1_file.name,
                'mean': None,
                'out': t1_out.name
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

    def test_FunManager(self):

        args = {'min':[[0,4]],
                'smin':[[0,2,4]],
                'ids':[[1]],
                'indel':',',
                'outdel':None}
        parg = pygby.Parser().parse_args(args)
        self.assertEqual(parg['min'], (0,2))
        self.assertEqual(set(parg['smin'][0].keys()), {'by', 'record'})
        self.assertEqual(set(parg['smin'][0].values()), {0, (1,2)})
        # Obligate floats (involved in numerical calculations)
        self.assertEqual(parg['floats'], {0,4})
        # All indices used in data row
        self.assertEqual(parg['allids'], {0,2,4})

        args = {'min':[[0,4],[0,5]],
                'smin':[[0,2,4],[5,7,8],[5,9]],
                'ids':[[1,2]],
                'indel':',',
                'outdel':None}
        parg = pygby.Parser().parse_args(args)
        self.assertEqual(parg['min'], (0,1,2))
        self.assertEqual(set(parg['smin'][1].keys()), {'by', 'record'})
        self.assertEqual(set(parg['smin'][1].values()), {2, (3,4,5)})
        # Obligate floats (involved in numerical calculations)
        self.assertEqual(parg['floats'], {0,4,5})
        # All indices used in data row (2 should not be here)
        self.assertEqual(parg['allids'], {0,4,5,7,8,9})

if __name__ == '__main__':
    unittest.main()
