__author__ = 'casey'


def get_overlap(first, second):
    if first is None and second is None:
        return None
    elif first is None and second is not None:
        return second
    elif first is not None and second is None:
        return first
    else:
        if isinstance(first, (tuple, list, set)) and isinstance(second, (tuple, list, set)) \
                and len(first) == 2 and len(second) == 2:
            unset = '__unset__'

            low = _get_overlap_nones(first[0], second[0], unset)
            high = _get_overlap_nones(first[1], second[1], unset)
            if low != unset and high != unset:
                return (low, high)
            if None not in first and first[0] > first[1]:
                raise AttributeError("Argument tuples must be sorted.")
            if None not in second and second[0] > second[1]:
                raise AttributeError("Argument tuples must be sorted.")

            if None in first or None in second:
                low = unset
                high = unset
                if first[0] is None and second[0] is None:
                    low = None
                elif first[0] is None:
                    low = second[0]
                elif second[0] is None:
                    low = first[0]
                else:
                    low = max(first[0], second[0])

                if first[1] is None and second[1] is None:
                    high = None
                elif first[1] is None:
                    high = second[1]
                elif second[1] is None:
                    high = first[1]
                else:
                    high = min(first[1], second[1])

                return tuple((low, high))

            elif first[0] > first[1] or second[0] > second[1]:
                raise AttributeError("Argument tuples must be sorted.")
            elif first[0] <= second[0]:
                if first[1] < second[0]:
                    raise RuntimeError('No overlap')
                elif first[1] >= second[1]:
                    return second
                else:
                    return tuple((second[0], first[1]))
            else:
                if first[0] > second[1]:
                    raise RuntimeError('No overlap')
                elif first[1] <= second[1]:
                    return first
                else:
                    return tuple((first[0], second[1]))
        else:
            raise TypeError("Invalid arguments types %s type %s and %s type %s" % (first, type(first), second, type(second)))


def _get_overlap_nones(a,b,unset):
    ret = None
    if a is None and b is None:
        ret = None
    elif a is None:
        ret = b
    elif b is None:
        ret = a
    else:
        ret = unset
    return ret