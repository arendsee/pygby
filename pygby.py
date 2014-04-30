#!/usr/bin/env python3

import csv
import sys
import argparse
from itertools import chain

__version__ = '1.0'
__prog__ = 'pygby'

class FunMap:
    def _sd():
        '''
        Prepares an anonymous standard deviation calculating function

        @return: Standard deviation function
        @rtype: function
        '''
        def sd(x):
            '''
            Prepares unbiased standard deviation calculating function. If input
            is less than 2, it returns 'NA'.

            @param x: vector of floats
            @type x: iterable<float>
            @return: The standard deviation of the input vector
            @rtype: float
            '''
            if(len(x) < 2):
                return('NA')
            else:
                mean = sum(x) / len(x)
                stdev = (sum((y - mean) ** 2 for y in x) / (len(x) - 1)) ** 0.5
                return(stdev)
        return(sd)

    def _median():
        '''
        Prepares an anonymous median calculating function

        @return: median function
        @rtype: function
        '''
        def median(x):
            '''
            Prepares median calculating function. If vector length is even,
            returns mean of innermost two values.

            @param x: vector of floats
            @type x: iterable<float>
            @return: The median of the input vector
            @rtype: float
            '''
            half = len(x) // 2
            isodd = len(x) % 2 != 0
            x = sorted(x)
            return(x[half] if isodd else sum(x[half-1:half+1])/2)
        return(median)

    numeric = ('max', 'min', 'mean', 'median', 'sd', 'sum')
    select = ('smax', 'smin')
    funmap = {
        'max':max,
        'min':min,
        'smax':max,
        'smin':min,
        'sum':sum,
        'mean':lambda x: sum(x) / len(x),
        'median':_median(),
        'sd':_sd(),
    }

class FunManager:
    '''
    Input class parses command line arguments and prepares data manipulation
    functions. Does NOT open the input file or have ANY knowledge of header
    content. Will know which indices are ID columns and which columns undergo
    manipulations.
    '''
    def __init__(self, args):
        self._args = args
        self.header = args['header']
        self.ids = sorted(args['ids'])
        self.datids = sorted(args['allids'].difference(args['ids']))
        # List of functions to perform on data row to make output row
        self._data_fun = []
        # List of functions to perform on data rownames to make header
        self._head_fun = []
        # Make _data_fun and _head_fun
        self._make_functions()

    def _make_functions(self):
        df = []
        nf = []
        # Get function map: dict<str:function>
        fun = FunMap.funmap
        # Filter to get arguments from a list that exist
        given = lambda x: [f for f in x if self._args[f]]

        # Append simple numeric functions
        for arg in given(FunMap.numeric):
            for i in self._args[arg]:
                df.append(lambda x,i=i,f=arg: (fun[f]([y[i] for y in x]),))
                nf.append(lambda x,i=i,f=arg: tuple(['%s.%s' % (x[i], f)]))

        def create_sel(f,d):
            def sel(x):
                byrow = tuple(y[d['by']] for y in x)
                selid = byrow.index(fun[f](byrow))
                selcols = tuple(x[selid][i] for i in d['record'])
                return(selcols)
            return(sel)

        # Append select functions
        for arg in given(FunMap.select):
            for d in self._args[arg]:
                df.append(create_sel(arg,d))
                s = '%s.where.%s.%s'
                nf.append(lambda x,f=arg,b=d['by'],rec=d['record']: \
                          [s % (x[r], x[b], f) for r in rec])

        # Append count if user desires
        if self._args['count']:
            df.append(lambda x: len(x))
            nf.append(lambda x: 'count')

        self._data_fun = df
        self._head_fun = nf

    def get_reader(self):
        '''
        Prepare a csv reader object

        @return: a csv reader according to user input
        @rtype: csv.reader
        '''
        reader = csv.reader(self._args['in'], delimiter=self._args['outdel'])
        return(reader)

    def get_writer(self):
        '''
        Prepare a csv writer object

        @return: a csv writer according to user input
        @rtype: csv.writer
        '''
        writer = csv.writer(self._args['out'], delimiter=self._args['outdel'])
        return(writer)

    def get_datarow(self, row):
        '''
        This function is the bottleneck
        '''
        try:
            for i in self._args['floats']:
                row[i] = float(row[i])
            row = tuple(row[i] for i in self.datids)
        except ValueError:
            msg = "Can't perform numeric function on column containing non-numeric data"
            print(msg, file=sys.stderr)
            raise SystemExit
        return(row)

    def get_outrow(self, ids, dat):
        row = tuple(chain(ids, *(f(dat) for f in self._data_fun)))
        return(row)

    def get_out_header(self, reader):
        header = [list(reader.idnames)] + \
                 [f(reader.colnames) for f in self._head_fun]
        row = [x for x in chain(*header)]
        return(row)

class Reader:
    '''
    Reads input file.
    @param header: is the first row in the input a header?
    @type header: bool
    @param kwargs: Arguments to pass to csv.reader
    @type kwargs: dict
    '''
    def __init__(self, funman):
        # All input data ((id1, id2, ...), ((col1, col2, ...),(...),..))
        self.data = []
        # Tuple of idnames ordered as in intput
        self.idnames = None
        # Tuple of non-id names ordered as in input
        self.colnames = None
        self._read(funman)

    def _read(self, funman):
        csvreader = funman.get_reader()

        if(funman.header):
            first = next(csvreader)
            self.idnames = tuple(first[i] for i in funman.ids)
            self.colnames = tuple(first[i] for i in funman.datids)
        else:
            self.idnames = tuple('Col%d' % i for i in funman.ids)
            self.colnames = tuple('Col%d' % i for i in funman.datids)

        # Iterate through the rows assigning data columns to id keys
        data = []
        for row in csvreader:
            data.append((tuple(row[i] for i in funman.ids), funman.get_datarow(row)))

        def tuplend(x):
            x[-1] = (x[-1][0], tuple(tuple(y) for y in x[-1][1]))
        data = sorted(data, key=lambda x: x[0])
        self.data = [[data[0][0], []]]
        for row in data:
            if(row[0] == self.data[-1][0]):
                self.data[-1][1].append(row[1])
            else:
                tuplend(self.data)
                self.data.append([row[0], [row[1]]])
        tuplend(self.data)

class Parser:
    '''
    Parse command line arguments

    @return: all user outputs retrieved by argparse
    @rtype : dict<str:?>
    '''
    def _reindex_args(self, ids2remove, args):
        '''
        Removes indices from vector and reindex value to key-less vector value

        @param values: input fields corresponding to csv columns
        @type values: iterable
        @param ids: group-by column indices
        @type ids: iterable
        @return: set<int>
        '''
        def reindex(v):
            return(v - sum([v > i for i in ids2remove]))

        def keep(x):
            return(tuple(reindex(y) for y in x if y not in ids2remove))

        for arg in args:
            if not args[arg]:
                continue
            if(arg in FunMap.numeric):
                args[arg] = keep(args[arg])
            if(arg in FunMap.select):
                for d in args[arg]:
                    d['by'] = reindex(d['by'])
                    d['record'] = keep(d['record'])

    def _get_args(self):
        parser = argparse.ArgumentParser(
            prog=__prog__,)
        parser.add_argument(
            '--version',
            help='Display version',
            action='version',
            version='%(prog)s {}'.format(__version__)
        )
        # Type: readable
        parser.add_argument(
            '-i', '--in',
            help='Input csv file (default: stdin)',
            nargs='?',
            type=argparse.FileType('r'),
            default=sys.stdin
        )
        # Type: writable
        parser.add_argument(
            '-o', '--out',
            help='Output csv file (default: stdin)',
            nargs='?',
            type=argparse.FileType('w'),
            default=sys.stdout
        )
        # Type: bool
        parser.add_argument(
            '--header', dest='header',
            help="First row is a header (one name per column)",
            action="store_true",
            default=False
        )
        parser.add_argument(
            '--count', dest='count',
            help="Count number of groups",
            action="store_true",
            default=False
        )
        # Type: str
        parser.add_argument(
            '-d', '--in-delimiter', dest='indel', metavar='in-delimiter',
            help='Column delimiter (default=TAB)',
            default='\t'
        )
        parser.add_argument(
            '--out-delimiter', dest='outdel', metavar='out-delimiter',
            help='Column output delimiter (default=in-delimiter)'
        )
        # Type: list<list<int>>
        parser.add_argument(
            '-g', '--group-by', dest='ids', metavar='ids',
            help='Indices by which to group (default=0)',
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--smax', dest='smax', metavar='smax',
            help='Select row elements where column is maximal (ARG1 = max column, ARG2+ = printed columns)',
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--smin', dest='smin', metavar='smin',
            help='Like kmax, but with minimal values',
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--max', dest='max', metavar='max',
            help="Select rows with maximal values in numeric columns",
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--min', dest='min', metavar='min',
            help="Select minimal values in given numeric columns",
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--sum', dest='sum', metavar='sum',
            help="Sum across given numeric columns",
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--mean', dest='mean', metavar='mean',
            help="Calculate mean across given numeric columns",
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--median', dest='median', metavar='median',
            help="Calculate median across given numeric columns",
            type=int,
            action='append',
            nargs='+'
        )
        parser.add_argument(
            '--sd', dest='sd', metavar='sd',
            help="Calculate standard deviation across given numeric columns",
            type=int,
            action='append',
            nargs='+'
        )

        if(len(sys.argv) == 1):
            parser.print_help()
            raise SystemExit

        return(vars(parser.parse_args()))

    def parse_args(self, args=None):
        from collections import defaultdict
        if not args:
            args = self._get_args()
        try:
            args['ids'] = {x for x in chain(*args['ids'])}
        except:
            args['ids'] = {0}
        args['floats'] = set()
        args['allids'] = set()
        # If no output delimiter is chosen, set to input delimiter
        if not args['outdel']:
            args['outdel'] = args['indel']
        for k,v in args.items():
            # Columns that will enter simple functions: convert to sets
            if(k in FunMap.numeric and v):
                args[k] = tuple(sorted(set(chain(*v))))
                args['floats'].update(args[k])
            # Selection based entries, convert to dicts with 'by' as key and
            # 'selected' as values
            if(k in FunMap.select and v):
                sel = defaultdict(set)
                for w in v:
                    # If the only value is an id, ignore the stupid user
                    if(all([x in args['ids'] for x in w[1:]]) or w[0] in args['ids']):
                        continue
                    args['floats'].update([w[0]])
                    args['allids'].update(w)
                    sel[w[0]].update(w[1:])
                keys = sorted(sel.keys())
                args[k] = tuple({'by':j, 'record':tuple(sorted(sel[j]))} for j in keys)
            args['allids'].update(args['floats'])
            args['allids'] = args['allids'] - set(args['ids'])
        ids2remove = args['ids'] | set(range(max(args['allids']) + 1)) - args['allids']
        self._reindex_args(ids2remove, args)
        return(args)

def write():
    args = Parser().parse_args()
    funman = FunManager(args)
    reader = Reader(funman)
    writer = funman.get_writer()
    writer.writerow(funman.get_out_header(reader))
    for ids,dat in reader.data:
        writer.writerow(funman.get_outrow(ids, dat))

if __name__ == '__main__':
    write()
