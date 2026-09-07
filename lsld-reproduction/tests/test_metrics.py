from lsld_repro.metrics import summarize_measurements

def test_summary_reports_paired_distractor_shift():
    records = [{'relation': 'r', 'setting': 'related', 'gold_logit_no_context': 5.0, 'gold_logit_with_context': 5.5, 'distractor_logit_no_context': 1.0, 'distractor_logit_with_context': 3.0, 'distractor_probability_no_context': 0.01, 'distractor_probability_with_context': 0.03}, {'relation': 'r', 'setting': 'related', 'gold_logit_no_context': 4.0, 'gold_logit_with_context': 4.25, 'distractor_logit_no_context': 0.0, 'distractor_logit_with_context': 1.0, 'distractor_probability_no_context': 0.02, 'distractor_probability_with_context': 0.04}]
    summary = summarize_measurements(records)[0]
    assert summary['n'] == 2
    assert summary['distractor_logit_delta'] == 1.5
    assert summary['gold_logit_delta'] == 0.375
    assert abs(summary['distractor_relative_probability_delta'] - 1.5) < 1e-12
