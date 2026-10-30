"""Past-only MVDR covariance and arbitrary-chunk numerical checks."""

import numpy as np
import pytest

from experiments.probe_causal_mvdr import past_mvdr


def test_no_future_frames_and_fixed_covariance_memory():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(31,7,4))+1j*rng.normal(size=(31,7,4))
    mask=rng.uniform(0,1,size=(31,7))
    full,_=past_mvdr(x,mask)
    changed=x.copy();changed[13:]*=7
    other_mask=mask.copy();other_mask[13:]=1-other_mask[13:]
    other,_=past_mvdr(changed,other_mask)
    state=None;parts=[]
    for a,b in ((0,1),(1,8),(8,19),(19,31)):
        result,state=past_mvdr(x[a:b],mask[a:b],state)
        parts.append(result)
        assert all(value.shape==(1,7,4,4) for value in state.values())
    for name,value in full.items():
        np.testing.assert_array_equal(value[:13],other[name][:13])
        np.testing.assert_allclose(value,np.concatenate([part[name] for part in parts]),rtol=1e-11,atol=1e-11)


def test_silence_is_finite_without_a_reference_signal():
    result,_=past_mvdr(np.zeros((7,5,8),complex),np.zeros((7,5)))
    assert all(np.isfinite(value).all() and np.count_nonzero(value)==0 for value in result.values())
    with pytest.raises(ValueError):past_mvdr(np.full((7,5,8),np.nan),np.zeros((7,5)))
