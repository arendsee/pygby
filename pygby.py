#!/usr/bin/env python3

import csv
import sys
import argparse
from itertools import chain

__version__ = 'development'
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
        self.silent_header = args['silent_header']
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
            df.append(lambda x: [len(x)])
            nf.append(lambda x: ['count'])

        self._data_fun = df
        self._head_fun = nf

    def get_reader(self):
        '''
        Prepare a csv reader object

        @return: a csv reader according to user input
        @rtype: csv.reader
        '''
        reader = csv.reader(self._args['in'], delimiter=self._args['indel'])
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

        def indexErrorReport(ncol):
            print('You request action on column %d, but the data '
                  'appear to have only %d column(s)' % \
                  (max(funman.ids + funman.datids) + 1, ncol),
                  file=sys.stderr)
            raise SystemExit

        if(funman.header):
            try:
                first = next(csvreader)
                self.idnames = tuple(first[i] for i in funman.ids)
                self.colnames = tuple(first[i] for i in funman.datids)
            except IndexError:
                indexErrorReport(len(first))
        else:
            self.idnames = tuple('Col%d' % int(i + 1) for i in funman.ids)
            self.colnames = tuple('Col%d' % int(i + 1) for i in funman.datids)

        # Iterate through the rows assigning data columns to id keys
        data = []
        try:
            for row in csvreader:
                data.append((tuple(row[i] for i in funman.ids), funman.get_datarow(row)))
        except IndexError:
            indexErrorReport(len(row))

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

    def _get_args(self, arglist=None):
        class ListAction(argparse.Action):
            def __init__(self,
                        option_strings,
                        dest,
                        default=None,
                        help=None,
                        metavar=None):
                super(ListAction, self).__init__(
                    option_strings=option_strings,
                    dest=dest,
                    nargs='+',
                    const=None,
                    default=default,
                    type=int,
                    choices=None,
                    required=False,
                    help=help,
                    metavar=metavar)

            def tozero(self, values):
                '''
                Converts 1-based vectors to 0-base vectors
                '''
                # Check all indices are greater than 0
                if(any([v < 1 for v in values])):
                   print('Column indices must be greater than 0 (numbering begins at 1 not 0)',
                         file=sys.stderr)
                   raise SystemExit
                # Convert to 0-based indices
                return([v - 1 for v in values])

            def getprior(self, namespace):
                '''
                Retrieves input from prior argument (e.g. pygby --max 8 --max 2 3)
                '''
                from copy import copy
                prior = copy(argparse._ensure_value(namespace, self.dest, []))
                return(prior)

        class ColumnList(ListAction):
            '''
            Converts input to 0-based list of sorted unique integers
            '''
            def __call__(self, parser, namespace, values, option_string=None):
                values = self.tozero(values)
                prior = set(self.getprior(namespace))
                prior.update(values)
                prior = sorted(prior)
                setattr(namespace, self.dest, prior)

        class SelectList(ListAction):
            def __call__(self, parser, namespace, values, option_string=None):
                '''
                Converts input to list<dict> where dicts are of form:
                    {'by':<int>, 'record':list<int>>}
                - 'by' == values[0], 'record' == values[1:]
                - 'record' list includes sorted unique values
                - all integers are converted to 0-base
                - identical 'by' entries are merged
                '''
                if len(values) < 2:
                    print('Select options (smin and smax) require at least 2 options:\n'
                          'arg1 determines which row to keep, arg2+ determines which'
                          'elements to record',
                          file=sys.stderr)
                    raise SystemExit
                values = self.tozero(values)
                prior = self.getprior(namespace)
                d = {'by':values[0], 'record':sorted(values[1:])}
                for p in prior:
                    if p['by'] == d['by']:
                        p['record'] += d['record']
                        p['record'] = sorted(set(p['record']))
                        setattr(namespace, self.dest, prior)
                        return
                prior.append(d)
                setattr(namespace, self.dest, prior)

        parser = argparse.ArgumentParser(
            prog=__prog__,
            description='Group csv file by given columns and report various info about other columns'
        )
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
            metavar='str',
            type=argparse.FileType('r'),
            default=sys.stdin
        )
        # Type: writable
        parser.add_argument(
            '-o', '--out',
            help='Output csv file (default: stdin)',
            nargs='?',
            metavar='str',
            type=argparse.FileType('w'),
            default=sys.stdout
        )
        # Type: bool
        parser.add_argument(
            '--header', dest='header',
            help="First row of input is a header",
            action="store_true",
            default=False
        )
        parser.add_argument(
            '--count', dest='count',
            help="Count number of groups",
            action="store_true",
            default=False
        )
        parser.add_argument(
            '--silent-header', '-s',
            help='Do not print header',
            action='store_true',
            default=False
        )
        # Type: str
        parser.add_argument(
            '-d', '--in-delimiter', dest='indel', metavar='int',
            help='Column delimiter (default=TAB)',
            default='\t'
        )
        parser.add_argument(
            '--out-delimiter', dest='outdel', metavar='int',
            help='Column output delimiter (default=in-delimiter)'
        )
        # Type: list<list<int>>
        parser.add_argument(
            '-g', '--group-by', dest='ids', metavar='int',
            help='Indices by which to group (default=0)',
            action=ColumnList,
        )
        parser.add_argument(
            '--smax', dest='smax', metavar='int',
            help='Select row elements where column is maximal (ARG1 = max column, ARG2+ = printed columns)',
            action=SelectList,
        )
        parser.add_argument(
            '--smin', dest='smin', metavar='int',
            help='Like kmax, but with minimal values',
            action=SelectList,
        )
        parser.add_argument(
            '--max', dest='max', metavar='int',
            help="Select maximal value in given numeric columns",
            action=ColumnList,
        )
        parser.add_argument(
            '--min', dest='min', metavar='int',
            help="Select minimal value in given numeric columns",
            action=ColumnList,
        )
        parser.add_argument(
            '--sum', dest='sum', metavar='int',
            help="Sum across given numeric columns",
            action=ColumnList,
        )
        parser.add_argument(
            '--mean', dest='mean', metavar='int',
            help="Calculate mean across given numeric columns",
            action=ColumnList,
        )
        parser.add_argument(
            '--median', dest='median', metavar='int',
            help="Calculate median across given numeric columns",
            action=ColumnList,
        )
        parser.add_argument(
            '--sd', dest='sd', metavar='int',
            help="Calculate standard deviation across given numeric columns",
            action=ColumnList,
        )

        if(len(sys.argv) == 1 and not arglist):
            parser.print_help()
            raise SystemExit

        args = vars(parser.parse_args(arglist))

        return(args)

    def parse_args(self, arglist=None):
        '''
        - sets argument defaults where needed
        - identify columns that undergo numerical operations for subsequent
        conversion to floats.
        - identify all columns that are used, unused columns can be ignored,
        thus improving memory efficiency

        Internally each line of input data is converted to two tuples: an id
        tuple and a data tuple. The user-provided data ids and group ids are
        converted to indices in these tuples.

        Example:
            pygby --group-by 1 5 --min 2 8 9 --max 6
        argparse yields:
            args.min -> [1,7,8]
            args.ids -> [0,4]
            args.max -> [5]
        this function converts these to:
            - Map to internal tuples
            args['min'] -> [0,2,3]
            args['max'] -> [1]
            args['ids'] -> [0,1]
            - Map to external csv columns
            args['floats'] -> [1,5,7,8]
            args['allids'] -> [0,1,4,5,7,8]
        '''
        args = self._get_args(arglist)

        if not args['ids']:
            args['ids'] = [0]

        # If no output delimiter is chosen, set to input delimiter
        if not args['outdel']:
            args['outdel'] = args['indel']

        args['floats'] = set()
        args['allids'] = set()
        for k,v in args.items():
            if k in FunMap.numeric and v:
                args['floats'].update(args[k])
            # Selection based entries, convert to dicts with 'by' as key and
            # 'selected' as values
            if k in FunMap.select and v:
                for w in v:
                    # If by is an id column or if all the recordings are ids,
                    # there is nothing to report, so delete item
                    if w['by'] in args['ids'] or all([x in args['ids'] for x in w['record']]):
                        del v[v.index(w)]
                        continue
                    args['floats'].update([w['by']])
                    args['allids'].update(w['record'])
            args['allids'].update(args['floats'])
        try:
            args['allids'] = args['allids'] - set(args['ids'])
            ids2remove = set(args['ids']) | set(range(max(args['allids']) + 1)) - args['allids']
            self._reindex_args(ids2remove, args)
        except ValueError:
            # If this error occurs, then there are no data columns. This is fine.
            pass
        return(args)


def write(arglist=None, returnlist=False):
    args = Parser().parse_args(arglist)
    funman = FunManager(args)
    reader = Reader(funman)

    def row_iter():
        if not funman.silent_header:
            yield funman.get_out_header(reader)
        for i,d in reader.data:
            yield funman.get_outrow(i, d)

    if returnlist:
        return([list(r) for r in row_iter()])
    else:
        funman.get_writer().writerows(row_iter())

if __name__ == '__main__':
    write()
