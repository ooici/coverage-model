from basic_types import AxisTypeEnum, MutabilityEnum, VariabilityEnum
from coverage import SimplexCoverage, SimpleDomainSet, GridDomain, GridShape, CRS
from parameter import ParameterContext, ParameterDictionary
from parameter_types import ArrayType, BooleanType, CategoryRangeType, CategoryType, ConstantType, CountRangeType, CountType, FunctionType, QuantityRangeType, QuantityType, RecordType, ReferenceType, TextType, TimeRangeType, TimeType, VectorType, ConstantRangeType
from parameter_values import get_value_class
from utils import create_guid, fix_slice
from numexpr_utils import make_range_expr

_core = ['ParameterContext', 'ParameterDictionary', 'SimplexCoverage', 'SimpleDomainSet',
         'GridDomain', 'GridShape', 'CRS', 'AxisTypeEnum', 'MutabilityEnum', 'VariabilityEnum']

_types = ['ArrayType', 'BooleanType', 'CategoryRangeType', 'CategoryType', 'ConstantType',
          'CountRangeType', 'CountType', 'FunctionType', 'QuantityRangeType', 'QuantityType',
          'RecordType', 'ReferenceType', 'TextType', 'TimeRangeType', 'TimeType', 'VectorType', 'ConstantRangeType']

_utils = ['make_range_expr', 'create_guid', 'get_value_class', 'fix_slice']

# Determines the set of things imported by using:  from coverage_model import *
__all__ = _core + _types + _utils