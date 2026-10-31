"""Screen current-frame global frequency attention using the same immutable eight-crop control."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN
import experiments.probe_causal_pmsqe as probe


class SourceCompatibleAttention(CausalFrequencyAttentionTCN):
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Both source and control lack only the seven newly initialized frequency-attention tensors.
        result = super().load_state_dict(state_dict, strict=False, **kwargs)
        assert not result.unexpected_keys, result.unexpected_keys
        assert all(name.startswith('mapping_input.frequency_') for name in result.missing_keys), result.missing_keys
        return result


def main():
    assert probe.sha(probe.__file__) == '5558a0bf7ad478180c32945775b6e984dbc28ef24c89ea21c007323d448cb9d9'
    implementation = {name:probe.sha(Path(__file__).with_name(name)) for name in
                      ('probe_causal_frequency_attention.py', 'causal_frequency_attention.py')}
    print(json.dumps(dict(architecture='CausalFrequencyAttentionTCN', additional_implementation_sha256=implementation)), flush=True)
    # Patch only this new process; the running single-change experiment is untouched.
    probe.CausalTCNMapping = SourceCompatibleAttention
    probe.main()
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1]) / 'probe.json'
    result = json.loads(output.read_text())
    assert result['status'] == 'complete' and result['parameters'] == result['trainable_parameters'] == 2574354
    result.update(architecture='CausalFrequencyAttentionTCN',
        scope='Global current-frame frequency attention with unchanged PMSQE loss; same8 training crops and200 updates; not generalization',
        additional_implementation_sha256=implementation)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
