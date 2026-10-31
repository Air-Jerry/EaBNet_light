"""Causal cross-spectrum conditioning, fixed state and analytic steering checks."""
import numpy as np
import pytest

from experiments.probe_causal_cross_mvdr import cross_mvdr


def test_cross_covariance_has_no_future_and_bounded_chunk_state():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(31,5,8))+1j*rng.normal(size=(31,5,8))
    s=x[...,0]*.3
    full,_=cross_mvdr(x,s)
    changed=x.copy();changed[13:]*=11
    other=s.copy();other[13:]*=3j
    result,_=cross_mvdr(changed,other)
    np.testing.assert_array_equal(full[:13],result[:13])
    state=None;parts=[]
    for a,b in ((0,1),(1,8),(8,19),(19,31)):
        value,state=cross_mvdr(x[a:b],s[a:b],state)
        parts.append(value)
        assert state['noise'].shape==(1,5,8,8) and state['cross'].shape==(1,5,8)
    np.testing.assert_allclose(np.concatenate(parts),full,rtol=1e-10,atol=1e-10)


def test_analytic_noiseless_steering_preserves_reference_zero():
    rng=np.random.default_rng(8)
    speech=rng.normal(size=(27,3))+1j*rng.normal(size=(27,3))
    steering=np.asarray([1, .7+.3j, -.2+.9j])
    x=speech[...,None]*steering
    actual,_=cross_mvdr(x,speech)
    np.testing.assert_allclose(actual,speech,rtol=1e-10,atol=1e-10)


def test_silent_estimate_falls_back_without_reference_or_nonfinite_state():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(7,5,8))+1j*rng.normal(size=(7,5,8))
    actual,_=cross_mvdr(x,np.zeros((7,5)))
    np.testing.assert_array_equal(actual,0)
    with pytest.raises(ValueError):cross_mvdr(x,np.full((7,5),np.nan))
    with pytest.raises(ValueError):cross_mvdr(x,np.ones((7,5)),state={'cross':np.zeros((1,5,9))})
