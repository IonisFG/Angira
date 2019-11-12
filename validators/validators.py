"""
Validators: angira_validators: add your own validator here
"""
__author__ = "Swagatam M"
__copyright__ = "@swagatam mukhopadhyay"
__credits__ = []
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Swagatam Mukhopadhyay"
__email__ = "smukhopadhyay@ionisph.com"
__status__ = "Dev"

import os, sys, string
import warnings
import re
import StringIO
import logging
from collections import defaultdict
import pandas as pd
import numpy as np

## LOCAL
from angira.searchers import searchers as search

######################################################################################

def _numcheckvector(vals, int_type =  False, val_range = None, val_set = None):
    """
    Basic validator for NUMERIC TYPE CHECKED values for checking range etc.
    Args:
    vals: NUMERIC TYPE CHECKED value array
        int_type: True, or False for float
        val_range: Can be None, check values in range
        val_set: Can be None, check values belong to set
    returns:
        bool index of valid array
    """
    vals = np.asarray(vals)
    assert not((val_set is not None) & (val_range is not None)), "Cannot have both num range as set of allowed nums: ambiguous"
    if int_type:
        check = np.equal(np.mod(vals, 1), 0) # 0 on modulo 1, so int
    else:
        check = np.ones(len(vals)).astype(bool)
    pass_inds = np.flatnonzero(check)
    if val_range is not None:
        L, R = val_range
        if (L is not None) & (R is not None):
            rangecheck = np.flatnonzero((vals[check] >= L) & (vals[check] <= R))
            pass_inds  = pass_inds[rangecheck]
        elif L is not None:
            rangecheck = np.flatnonzero((vals[check] >= L))
            pass_inds  = pass_inds[rangecheck]
        elif R is not None:
            rangecheck = np.flatnonzero((vals[check] <= R))
            pass_inds  = pass_inds[rangecheck]
        else:
            raise ValueError('bad range argument for string length')

    elif val_set is not None:
        valcheck = np.isin(vals[check], val_set)
        pass_inds  = pass_inds[np.flatnonzero(valcheck)]
    ans = np.zeros(len(vals)).astype(bool)
    ans[pass_inds] = True
    return ans

##########################################################################################


def _stringcheckvector(vals, val_set = None, string_lens_set = None, string_lens_range = None, token = None, search_func = None,
                       search_func_params = None):
    """
    Basic validator for STRING TYPAE CHECKED value entries
    Args:
        val_set: set of values, stripped
        string_lens_set = set of string length values
        string_lens_range = string length range as [L, R], [L, None] or [None, R] where L and R are integers
        token: a pattern match pattern or token digestible by searcher
        search_func: re.match.... fuzz.ratio etc. see searcher
        search_fucn_params: params for searcher function, see search class
    Returns:
        bool index of valid array
    """
    assert not(((string_lens_set is not None) & ((string_lens_range is not None)))), "Cannot have both string length range and length set: ambiguous"
    assert not(((token is not None) & ((val_set is not None)))), "Cannot have both pattern match and set of allowed values: superfluous"

    pass_inds1 = np.arange(len(vals)) #numeric index of array, need to do this because of ease of indexing later on. Boolean index array creates some complications.
    vals = np.asarray([v.strip() for v in vals])
    if string_lens_set is not None:
        lens = [len(v) for v in vals]
        pass_inds1 = np.flatnonzero(np.isin(lens, string_lens_set))
    elif string_lens_range is not None:
        lens = np.asarray([len(v) for v in vals])
        L, R =  string_lens_range
        if (L is not None) & (R is not None):
            pass_inds1 = np.flatnonzero((lens >= L) & (lens <= R))
        elif L is not None:
            pass_inds1 = np.flatnonzero(lens >= L)
        elif R is not None:
            pass_inds1 = np.flatnonzero(lens <= R)
        else:
            raise ValueError('bad range argument for string length')
    pass_inds2 = np.arange(len(vals)) #numeric index of array
    if val_set is not None:
        val_set =  [s.strip() for s in val_set]
        valcheck = np.isin(vals, val_set)
        pass_inds2 = np.flatnonzero(valcheck)
    elif token is not None:
        S = search.Search(token, search_func, params = search_func_params)
        matches = np.asarray([S(v) for v in vals[pass_inds2]])
        #Notice, this is tested on the subset that passed previous qualifiers
        pass_inds2 = pass_inds2[matches]
    pass_inds = set(pass_inds1).intersection(set(pass_inds2))
    ans = np.zeros(len(vals)).astype(bool)
    if len(pass_inds):
        ans[np.asarray(list(pass_inds))] = True
    return ans

####################################################################################################
############## These are semantic type checkers, some code resuse from validators ##################
####################################################################################################

def numTypeCheck(x, params = None, error_code = 'outRange'):
    """
    This is also a validator function with code reuse from NumValidator,
    but the purpose of this is to enable semantic type checks. It acts on
    vectors of values, instead of DataFrame
    Args:
        x: vector to validate
        params:
            int_type: True, or False for float
            val_range: Can be None, then just check whether it's int, check values belong to int
            val_set: Can be None, then  just check whether int, else check values belong to these
    Returns:
            indicator: True/False vector, same length as x
            errorVector: Error code for failure, same length as x
    """
    vals = np.asarray(x)
    assert np.ndim(vals) == 1, "Expect one dimensional list array"
    if params is None:
        _params = dict()
    else:
        _params = params.copy()
    _params.setdefault('int_type', False)
    _params.setdefault('val_set', None)
    _params.setdefault('val_range', None)
    #inds = ~pd.to_numeric(pd.Series(vals), errors  = 'coerce').isnull().values # NOT SAFE FOR DATETIME!!!
    inds = ~np.isnan(pd.to_numeric(vals, errors  = 'coerce'))
    failcode = np.asarray(['NotNum']*len(vals), dtype = 'S16')
    failcode[inds] = error_code
    ans = np.zeros(len(vals)).astype(bool)
    if np.any(inds):
        vals_loc = vals[inds].astype(float)
        pass_inds = _numcheckvector(vals_loc, **_params)
        z = np.flatnonzero(inds)[np.flatnonzero(pass_inds)]
        ans[z] = True
        failcode[z] = 'Pass'
    return ans, failcode


##########################################################################################

def stringTypeCheck(x, params = None, error_code = 'badStr'):
    """
    This is also a validator function with code reuse from StringValidator,
    but the purpose of this is to enable semantic type checks. It acts on
    vectors of values, instead of DataFrame
    Args:
        x: vector to validate
        params:
            val_set: set of values, stripped
            string_lens_set = set of string length values
            string_lens_range = string length range as [L, R], [L, None] or [None, R] where L and R are integers
            pattern: a pattern match, I will compile this using re.compile
    Returns:
            indicator: True/False vector, same length as x
            errorVector: Error code for failure, same length as x
    """
    vals = np.asarray(x)
    assert np.ndim(vals) == 1, "Expect one dimensional list array"
    if params is None:
        _params = dict()
    else:
        _params = params.copy()
    _params.setdefault('string_lens_set', None)
    _params.setdefault('string_lens_range', None)
    _params.setdefault('val_set', None)
    _params.setdefault('token', None)
    _params.setdefault('search_func', 're.match')
    inds = ~pd.Series(vals).isnull().values
    failcode = np.asarray(['NotStr']*len(vals), dtype = 'S16')
    failcode[inds] = error_code
    ans = np.zeros(len(vals)).astype(bool)
    if np.any(inds):
        vals_loc = vals[inds].astype(str)
        pass_inds = _stringcheckvector(vals_loc, **_params)
        ans = np.zeros(len(vals)).astype(bool)
        z = np.flatnonzero(inds)[np.flatnonzero(pass_inds)]
        ans[z] = True
        failcode[z] = 'Pass'
    return ans, failcode

###################################################################################################

def numUnitTypeCheck(x, params = None, error_code = 'badNumUnit'):
    """
    Validator for mixed types, i.e., numeric with units, for example, 1000 mg
    Args:
        x: vector to validate
        params:
            see NumTypeCheck and StringTypeCheck, for num_params and string_params
            see NumTypeCheck and StringTypeCheck, for num_params and string_params
            can pass pattern, defualt is '\s*([-+]?\d*\.\d+|\d+)\s*(\D*)\s*', i.e., number and unit, pattern
            must return two groups
            can pass order of index, as 'order': [0, 1] mean num first string second
            IF YOU CHNAGE THE PATTERN MAKE SURE TO PROVIDE THE CORRECT ORDER of sting and number extracted
    Returns:
            indicator: True/False vector, same length as x
            errorVector: Error code for failure, same length as x
    """
    vals = np.asarray(x)
    assert np.ndim(vals) == 1, "Expect one dimensional list array"
    assert params is not None, "need params with num_params and string_params defined"
    assert 'num_params' in params, "no num params"
    assert 'string_params' in params, "no unit params"
    num_params = params['num_params']
    string_params = params['string_params']
    params.setdefault('pattern', '\s*([-+]?\d*\.\d+|\d+)\s*(\D*)\s*') #num and unit
    params.setdefault('order', [0, 1])

    pattern = re.compile(params['pattern'])
    order = params['order']
    inds = ~pd.Series(vals).isnull().values
    failcode = np.asarray(['NotStr']*len(vals), dtype = 'S16')
    failcode[inds] = error_code
    ans = np.zeros(len(vals)).astype(bool)
    if np.any(inds):
        vals_loc = vals[inds].astype(str) # convert to string, otherwise re.match will complain if you passed numbers
        matches = np.asarray([True if re.match(pattern, str(v).strip()) is not None else False for v in vals_loc]) #pattern was found
        if np.any(matches):
            extracted = np.asarray([re.match(pattern, m.strip()).groups() for m in vals_loc[matches]])
            numbers = extracted[:, order[0]].astype(float) #this WILL error if the pattern you sent extratced strings in the first group
            units = np.asarray(map(string.strip, extracted[:, order[1]].astype(str)))
            pass_num_inds = _numcheckvector(numbers, **num_params)
            pass_str_inds = _stringcheckvector(units, **string_params)
            pass_inds = np.asarray(list(set(np.flatnonzero(pass_num_inds)).intersection(np.flatnonzero(pass_str_inds))))
            if len(pass_inds):
                z = np.flatnonzero(inds)[np.flatnonzero(matches)[pass_inds]]
                ans[z] = True
                failcode[z] = 'Pass'
    return ans, failcode
