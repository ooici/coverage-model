__author__ = 'casey'


def get_overlap(e1, e2):
    if e1 is None and e2 is None:
        return None
    elif e1 is None and e2 is not None:
        return e2
    elif e1 is not None and e2 is None:
        return e1
    else:
        if isinstance(e1, (tuple, list, set)) and isinstance(e2, (tuple, list, set)) \
                and len(e1) == 2 and len(e2) == 2\
                and e1[0] <= e1[1] and e2[0] <= e2[1]:
            if e1[0] <= e2[0]:
                if e1[1] < e2[0]:
                    raise RuntimeError('No overlap')
                elif e1[1] >= e2[1]:
                    return e2
                else:
                    return tuple((e2[0], e1[1]))
            else:
                if e1[0] > e2[1]:
                    raise RuntimeError('No overlap')
                elif e1[1] <= e2[1]:
                    return e1
                else:
                    return tuple((e1[0], e2[1]))
