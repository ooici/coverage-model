from basic_types import AxisTypeEnum, MutabilityEnum, VariabilityEnum
from coverage import SimplexCoverage, ViewCoverage, SimpleDomainSet, GridDomain, GridShape, CRS
from coverage_model.parameter_functions import PythonFunction, NumexprFunction
from parameter import ParameterContext, ParameterDictionary, ParameterFunctionValidator
from parameter_types import ArrayType, BooleanType, CategoryRangeType, CategoryType, ConstantType, CountRangeType, CountType, FunctionType, QuantityRangeType, QuantityType, RecordType, ReferenceType, TextType, TimeRangeType, TimeType, VectorType, ConstantRangeType, ParameterFunctionType
from parameter_values import get_value_class
from coverage_model import utils
from utils import create_guid, fix_slice
from numexpr_utils import make_range_expr
from coverage_model.base_test_cases import CoverageModelUnitTestCase, CoverageModelIntTestCase

_core = ['ParameterContext', 'ParameterDictionary', 'ParameterFunctionValidator', 'SimplexCoverage', 'ViewCoverage', 'SimpleDomainSet',
         'GridDomain', 'GridShape', 'CRS', 'AxisTypeEnum', 'MutabilityEnum', 'VariabilityEnum']

_types = ['ArrayType', 'BooleanType', 'CategoryRangeType', 'CategoryType', 'ConstantType',
          'CountRangeType', 'CountType', 'FunctionType', 'QuantityRangeType', 'QuantityType',
          'RecordType', 'ReferenceType', 'TextType', 'TimeRangeType', 'TimeType', 'VectorType', 'ConstantRangeType', 'ParameterFunctionType']

_functions = ['NumexprFunction', 'PythonFunction']

_utils = ['utils', 'make_range_expr', 'create_guid', 'get_value_class', 'fix_slice']

_test_cases = ['CoverageModelUnitTestCase', 'CoverageModelIntTestCase']

# Determines the set of things imported by using:  from coverage_model import *
__all__ = _core + _types + _utils + _functions + _test_cases