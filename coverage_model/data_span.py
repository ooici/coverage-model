__author__ = 'casey'

from ooi.logging import log

from address import Address, AddressFactory
from ast import literal_eval


class ParamSpan(object):

    @staticmethod
    def validate_param_value(key, val):
        if not isinstance(val, tuple) or not 2 == len(val) or not type(val[0]) == type(val[1]):
            raise ValueError("".join(
                ["params must be dict type with values a tuple of size 2 and both elements the same type.  Found ",
                 str(val), " for key ", key]))

    def __init__(self, address, params):
        if not isinstance(address, Address):
            raise ValueError("".join(['address must be Address type.  Found ', str(type(address))]))
        if not isinstance(params, dict):
            raise ValueError("".join(["params must be dict type.  Found ", str(type(params))]))

        self.address = address
        self.params = {}
        for key in params.keys():
            self.add_param(key, params[key])
        self.is_dirty = False
        super(ParamSpan, self).__setattr__('is_dirty', False)

    def __setattr__(self, key, value):
        super(ParamSpan, self).__setattr__(key, value)
        super(ParamSpan, self).__setattr__('is_dirty', True)

    def add_param(self, key, val):
        self.validate_param_value(key, val)
        if key in self.params:
            raise ValueError("".join(["key already exists in dictionary: ", key, " ", str(self.params[key])]))
        self.params[key] = val
        self.is_dirty = True

    def extend(self, span):
        self.extend_params(span.params)

    def extend_params(self, params):
        if not isinstance(params, dict):
            raise ValueError("".join(["params must be dict type.  Found ", str(type(params))]))
        for key in params.keys():
            self.extend_param(key, params[key])

    def extend_param(self, key, vals):
        if key not in self.params:
            self.add_param(key, vals)
        self.validate_param_value(key, vals)
        submitted_min = min(vals)
        submitted_max = max(vals)
        current_range = self.params[key]
        current_min = min(current_range)
        current_max = max(current_range)
        updated = False

        if submitted_min < current_min:
            current_min = submitted_min
            updated = True
        if submitted_max > current_max:
            current_max = submitted_max
            updated = True

        self.params[key] = (current_min, current_max)
        if updated:
            self.is_dirty = True

    def as_tuple(self):
        tup = [self.address.as_tuple()]
        for key in self.params.keys():
            tup.append((key, (self.params[key][0], self.params[key][1])))
        return tuple(tup)

    def __str__(self):
        return str(self.as_dict())

    def as_dict(self):
        rs = {'type': ParamSpan.__name__, 'address': str(self.address)}
        rs.update(self.params)
        return rs

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == ParamSpan.__name__:
            del dic['type']
            if 'address' in dic:
                address = AddressFactory.from_str(dic['address'])
                del dic['address']
                return ParamSpan(address, dic)
        raise ValueError("Do not know how to build ParamSpan from %s ", str(dic))

    @staticmethod
    def from_str(span_str):
        return ParamSpan.from_dict(literal_eval(span_str))

    @staticmethod
    def from_tuple(tup):
        address = ""
        params = {}
        skip = True
        for i in tup:
            if skip is True:
                address = AddressFactory.from_tuple(i)
                skip = False
                continue
            if isinstance(i, tuple) and len(i) == 2:
                params[i[0]] = i[1]
            else:
                raise ValueError("".join(
                    ["Unexpected tuple element.  Format is: ( name, (key, (val1,val2), (key, (val1,val2), ... Found ",
                     str(tup)]))
        if address is "":
            return None
        else:
            return ParamSpan(address, params)

    # Spans represent a set of parameter ranges for data at an address.
    # Sort them by address so we can identify and merge overlapping spans
    def __lt__(self, other):
        return self.address.__lt__ < other.address.__lt__()

    def __eq__(self, other):
        if self.address == other.address:
            return self.params == other.params
        return self.address == other.address

    def __ne__(self, other):
        return not self.__eq__(other)
        #if self.address == other.address:
        #    return self.params == other.params
        #return self.address == other.address


class SpanCollection(object):

    def __init__(self):
        # spans are stored as a dict of Span lists.
        # Address primary keys are used to map them to manageable collections. (Think brick/file pointer)
        self.span_dict = {}

    def add_span(self, span):
        address_key = span.address.get_top_level_key()
        if address_key in self.span_dict.keys():
            spans = self.span_dict[address_key]
            spans[str(span.address)] = span
            #TODO - Merge overlapping spans.
            # Current implementation has only one span per file so not necessary at this point
        else:
            tmp = dict()
            tmp[str(span.address)] = span
            self.span_dict[address_key] = tmp

    def get_span(self, address):
        if address.get_top_level_key() in self.span_dict:
            sub_spans = self.span_dict[address.get_top_level_key()]
            if str(address) in sub_spans:
                return sub_spans[str(address)]
        return None

    def get_dirty_spans(self):
        dirty_spans = []
        for key in self.span_dict.keys():
            spans = self.span_dict[key]
            for k2 in spans.keys():
                span = spans[k2]
                if span.is_dirty:
                    dirty_spans.append(span)
        return dirty_spans

    def as_dict(self):
        rs = {'type': SpanCollection.__name__}
        sp_d = {}
        for k, v in self.span_dict.iteritems():
            sp_d[k] = v.as_dict()
        rs['spans'] = sp_d
        return rs

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == SpanCollection.__name__:
            del dic['type']
            if 'spans' in dic and isinstance(dic['spans'], dict):
                span_dict = {}
                for k, v in dic['spans']:
                    if isinstance(v, dict):
                        span = ParamSpan.from_dict(v)
                        span_dict[k] = span
                return SpanCollection(span_dict=span_dict)
        raise ValueError("Do not know how to build SpanCollection from %s ", str(dic))

    @staticmethod
    def from_str(col_str):
        return SpanCollection(literal_eval(col_str))

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __str__(self):
        return str(self.as_dict())


class SpanCollectionByFile(object):
    def __init__(self):
        self.span_dict = {}

    def add_span(self, span):
        address_key = span.address.get_top_level_key()
        if address_key in self.span_dict.keys():
            log.debug("Extending span: %s", address_key)
            existing_span = self.span_dict[address_key]
            existing_span.extend(span)
        else:
            log.debug("Creating new span: %s", address_key)
            self.span_dict[address_key] = span

    def get_span(self, address):
        if address.get_top_level_key() in self.span_dict:
            return self.span_dict[address.get_top_level_key()]
        return None

    def get_dirty_spans(self):
        dirty_spans = []
        for key in self.span_dict.keys():
            span = self.span_dict[key]
            if span.is_dirty:
                dirty_spans.append(span)
        return dirty_spans

    def __str__(self):
        return str(self.as_tuple())

    def as_tuple(self):
        tup = [self.__class__.__name__]
        for key, span in self.span_dict.iteritems():
            if span is not None:
                tup.append(span.as_tuple())
                break
        return tuple(tup)

    @staticmethod
    def from_tuple_str(tup):
        collection = SpanCollectionByFile()
        skip = True
        for i in tup:
            if skip is True:
                if not str(i) == collection.__class__.__name__:
                    return i
                skip = False
                continue
            if isinstance(i, tuple):
                span = ParamSpan.from_tuple(i)
                collection.add_span(span)
            else:
                raise ValueError("".join(
                    ["Unexpected tuple element.  Format is: ( name, (key, (val1,val2), (key, (val1,val2), ... Found ",
                     str(tup)]))

        return collection

    def as_dict(self):
        rs = {'type': SpanCollectionByFile.__name__}
        sp_d = {}
        for k, v in self.span_dict.iteritems():
            sp_d[k] = v.as_dict()
        rs['spans'] = sp_d
        return rs

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == SpanCollectionByFile.__name__:
            del dic['type']
            if 'spans' in dic and isinstance(dic['spans'], dict):
                sc = SpanCollectionByFile()
                span_dict = {}
                for k in dic['spans'].keys():
                    v = dic['spans'][k]
                    if isinstance(v, str):
                        v = literal_eval(v)
                    if isinstance(v, dict):
                        span = ParamSpan.from_dict(v)
                        span_dict[k] = span
                        sc.add_span(span)
                return sc
        raise ValueError("Do not know how to build SpanCollectionByFile from %s ", str(dic))

    @staticmethod
    def from_str(col_str):
        return SpanCollectionByFile.from_dict(literal_eval(col_str))

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.as_dict())
