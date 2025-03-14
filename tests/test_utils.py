import pytest
from logdetective.utils import compute_certainty, initialize_model, process_log


@pytest.mark.parametrize("probs", (
    [{"logprob": 66.6}],
    [{"logprob": 99.9}, {"logprob": 1.0}]
))
def test_compute_certainty(probs):
    """ test compute_certainty and make sure we can use numpy correctly """
    compute_certainty(probs)


def test_process_log():
    # FIXME: parametrize this and mark as slow
    model = "Mungert/gemma-3-1b-it-gguf"
    suffix = "q4_k_s.gguf"
    model = initialize_model(model, filename_suffix=suffix, verbose=5)
    response = process_log("illegal hardware instruction (core dumped)", model, False)
    print(response)
