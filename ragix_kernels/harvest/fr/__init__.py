"""The deterministic French readers the harvest rests on: dates, typed values, cut values.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

No model reads a value here. Each reader returns exact spans of the text it was given,
typed and normalised where the writing allows it and incomplete with a reason where it
does not. `dates` reads dates and clock times, `grammars` the other critical kinds
(amounts, percentages, durations, references, quantities) and `cut` names the values a
text layer split across a line break, which a span check alone cannot see.
"""
