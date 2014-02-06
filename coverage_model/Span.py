__author__ = 'casey'

from ooi.logging import log

from address import Address
from ast import literal_eval


class Span(object):

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
        super(Span, self).__setattr__('is_dirty', False)

    def __setattr__(self, key, value):
        super(Span, self).__setattr__(key, value)
        super(Span, self).__setattr__('is_dirty', True)

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

    def as_tuple_str(self):
        tup = self.address.as_tuple_str(),
        for key in self.params.keys():
            tup = tup + ((key, (self.params[key])),)
        return tup

    def __str__(self):
        tup = self.address.as_tuple_str(),
        log.trace("Converting to string %s", str(tup))
        for key in self.params.keys():
            tup = tup + ((key, (self.params[key])),)
        return str(tup)

    @staticmethod
    def from_tuple_str(from_str):
        tup = literal_eval(from_str)
        address = ""
        params = {}
        skip = True
        for i in tup:
            if skip is True:
                address = Address.from_tuple_str(i)
                skip = False
                continue
            if isinstance(i, tuple) and len(i) == 2:
                params[i[0]] = i[1]
            else:
                raise ValueError("".join(
                    ["Unexpected tuple element.  Format is: ( name, (key, (val1,val2), (key, (val1,val2), ... Found ",
                     str(tup)]))
        return Span(address, params)

    # Spans represent a set of parameter ranges for data at an address.
    # Sort them by address so we can identify and merge overlapping spans
    def __lt__(self, other):
        return self.address.__lt__ < other.address.__lt__()


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


class SpanCollectionByFile(SpanCollection):
    def __init__(self):
        SpanCollection.__init__(self)

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
        out = []
        if len(self.span_dict) > 0:
            out.append(self.__class__.__name__)
            for key in self.span_dict.keys():
                span = self.span_dict[key]
                out.append(str(span))
        return str(tuple(out))

    @staticmethod
    def from_tuple_str(from_str):
        tup = literal_eval(from_str)
        collection = SpanCollectionByFile()
        skip = True
        for i in tup:
            if skip is True:
                if i is not SpanCollectionByFile.__class__.__name__:
                    return None
                skip = False
                continue
            if isinstance(i, basestring):
                log.trace('tuple string %s', i)
                span = Span.from_tuple_str(i)
                collection.add_span(span)
            else:
                raise ValueError("".join(
                    ["Unexpected tuple element.  Format is: ( name, (key, (val1,val2), (key, (val1,val2), ... Found ",
                     str(tup)]))

        return collection
