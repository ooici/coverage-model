#!/usr/bin/env python

if __name__ == "__main__":
    import os
    import shutil
    import re

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Remove coverages from the default or specified directory')
    parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
    parser.add_argument('-c', '--count', help='Report the number of coverages, but do not delete', action='store_true')
    parser.add_argument('loc', help='Location to remove coverages from', nargs='?', default=os.path.dirname(os.path.realpath(__file__)))

    guid_match = r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}'

    args = parser.parse_args()
    loc = os.path.realpath(args.loc)

    pths=[]
    for x in [x for x in os.listdir(loc) if re.match(guid_match, x) is not None]:
        pt=os.path.join(loc,x)
        if os.path.isdir(pt) and '{0}_master.hdf5'.format(x) in os.listdir(pt):
            pths.append(pt)
    
    nc = len(pths)
    if nc==0:
        print 'No coverages found in \'{0}\''.format(loc)
    else:
        if args.count:
            print '{0} coverages in directory \'{1}\''.format(nc, loc)            
        else:
            print 'Removing {0} coverages from \'{1}\'...'.format(nc, loc)
            for p in pths:
                if args.verbose:
                    print "Removing: %s" % p
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            print 'Finished!'



