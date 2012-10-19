#!/usr/bin/env python

if __name__ == "__main__":
    import sys
    import os
    import shutil
    import re

    guid_match = r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}'

    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        d = os.path.dirname(os.path.realpath(__file__))

    pths=[]
    for x in [x for x in os.listdir(d) if re.match(guid_match, x) is not None]:
        pt=os.path.join(d,x)
        if os.path.isdir(pt) and '{0}_master.hdf5'.format(x) in os.listdir(pt):
            pths.append(pt)
    
    if len(pths)==0:
        print 'No coverages found in {0}'.format(d)
    else:
        for p in pths:
            print "Removing: %s" % p
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)



