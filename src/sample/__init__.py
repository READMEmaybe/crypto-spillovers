"""Country-inclusion engine for the Step-2 panel.

The sample must be chosen on DATA QUALITY ALONE, blind to results. This package
audits per-country / per-channel data quality (`audit`), applies a declarative
rule frozen in `config/sample_rule.yaml` (`rule`), flags idiosyncratic domestic
crises for robustness (`crisis`), and renders an appendix table (`report`).
"""
