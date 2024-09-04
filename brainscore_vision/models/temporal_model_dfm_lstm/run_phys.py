from brainscore_vision import score
import os

os.environ["RESULTCACHING_DISABLE"] = "1"
#print(f"This script is running from: {__file__}")

def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score


actual_score = score(model_identifier="DFM-LSTM-ENCODER", benchmark_identifier="Physionv1.5-ocp", #"MajajHong2015public.IT-pls",
                         conda_active=True)
print(actual_score)
