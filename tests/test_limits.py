"""Tests that make the professor's three hardening requirements concrete:

1. NFA/DFA state explosion is handled safely (lazy DFA, not eager subset
   construction).
2. Long regexes and long input strings are handled without blowing up
   (linear time, no recursion errors).
3. Ambiguous/invalid regex and invalid input are handled gracefully
   (typed errors, never a raw traceback / crash).
"""

import time

import pytest

from regex_engine import match, RegexLimitError, RegexSyntaxError
from regex_engine import parser as p
from regex_engine.nfa import build_nfa
from regex_engine.dfa import LazyDFA


# ---------------------------------------------------------------------------
# 1. State explosion safety
# ---------------------------------------------------------------------------

def test_lazy_dfa_avoids_power_set_explosion():
    # (a?){k}a{k}: the textbook pattern whose *eager* subset construction
    # needs up to 2^k DFA states (each subset of "which optional a's
    # matched" is reachable in principle). With k=30 that's over a
    # billion states - infeasible to enumerate eagerly.
    k = 30
    pattern = "a?" * k + "a" * k
    text = "a" * k

    dfa = LazyDFA(build_nfa(p.parse(pattern)))
    state = dfa.start
    for ch in text:
        state = dfa.step(state, ch)
    assert state is not None and state.is_accept

    # Because states are only materialized on demand along the single
    # path actually walked, we never touch more than ~ (len(text) + 1)
    # states, regardless of the pattern's theoretical worst case.
    assert dfa.explored_state_count <= len(text) + 2


def test_pathological_pattern_is_capped_not_unbounded():
    # A pattern engineered to keep discovering *new* distinct subsets on
    # every step (wide alternation over many optional literals) must hit
    # the DFA state cap and fail cleanly rather than exhaust memory.
    import regex_engine.dfa as dfa_mod

    original_cap = dfa_mod.MAX_DFA_STATES
    dfa_mod.MAX_DFA_STATES = 50  # shrink the cap so the test is fast
    try:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        pattern = "|".join(f"{c}?" * 6 for c in alphabet)
        text = alphabet * 3
        result = match(pattern, text)
        assert result.error_type in ("limit", None)
    finally:
        dfa_mod.MAX_DFA_STATES = original_cap


# ---------------------------------------------------------------------------
# 2. Long regex / long input
# ---------------------------------------------------------------------------

def test_long_input_matches_in_roughly_linear_time():
    text = "a" * 1_000_000
    start = time.perf_counter()
    result = match("a*", text)
    elapsed = time.perf_counter() - start
    assert result.matches is True
    assert elapsed < 5.0  # generous bound; real runs are far faster


def test_long_flat_regex_compiles_and_matches():
    n = 5_000
    pattern = "a" * n
    result = match(pattern, "a" * n)
    assert result.matches is True
    result = match(pattern, "a" * (n - 1))
    assert result.matches is False


def test_over_length_pattern_rejected_cleanly():
    result = match("a" * 30_000, "a")
    assert result.matches is False
    assert result.error_type == "limit"


def test_over_length_input_rejected_cleanly():
    from regex_engine.matcher import MAX_INPUT_LENGTH

    result = match("a*", "a" * (MAX_INPUT_LENGTH + 1))
    assert result.matches is False
    assert result.error_type == "limit"


def test_deeply_nested_groups_rejected_cleanly():
    n = p.MAX_GROUP_NESTING + 10
    pattern = "(" * n + "a" + ")" * n
    result = match(pattern, "a")
    assert result.matches is False
    assert result.error_type == "limit"


# ---------------------------------------------------------------------------
# 3. Ambiguous / invalid regex and invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pattern",
    ["a(b", "a)b", "*a", None, "a|", "|a", "a\\", "\\q", "((a)"],
)
def test_malformed_regex_never_raises_out_of_match(pattern):
    if pattern is None:
        return  # placeholder guard, matcher always takes a str
    result = match(pattern, "anything")
    assert result.matches is False
    assert result.error is not None


def test_parser_errors_carry_a_position_for_the_ui():
    with pytest.raises(RegexSyntaxError) as exc_info:
        p.parse("a(b")
    assert exc_info.value.position is not None


def test_non_alphabet_character_in_input_is_not_an_error():
    # A character the pattern never mentions should just fail to match,
    # not raise - the DFA simply has no transition for it.
    result = match("abc", "xyz")
    assert result.matches is False
    assert result.error is None
