import os
from brainscore_vision import score

os.environ["RESULTCACHING_DISABLE"] = "1"
#print(f"This script is running from: {__file__}")

def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score


actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-ocp", #"MajajHong2015public.IT-pls",
                         conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-ocd", #"MajajHong2015public.IT-pls",
                                 conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-snippet-rollout-performance", #"MajajHong2015public.IT-pls",
                                 conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-snippet-rollout-intra-performance", #"MajajHong2015public.IT-pls",
                                 conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-ocp-intra-generalization", #"MajajHong2015public.IT-pls",
                                 conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-ocd-intra-generalization", #"MajajHong2015public.IT-pls",
                                         conda_active=True)

actual_score = score(model_identifier="DFM-LSTM-SIM", benchmark_identifier="Physionv1.5-snippet-simulation-performance", #"MajajHong2015public.IT-pls",
                                         conda_active=True)

print(actual_score)
