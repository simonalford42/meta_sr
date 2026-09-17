# Synthetic SRBench2 portfolio plot

Source: `runs/srbench2_9-17_1m_spliced_terra_recovery`.

New first restarts replace archived first restarts; later fits are reused and timestamps shifted. Solid lines connect cumulative recovery values at the shared checkpoints, including checkpoints with no increase. Five-second captures use their scheduled times (5, 10, 15 seconds, etc.), not the slightly delayed actual read times. The common checkpoint grid combines scheduled captures and recorded restart-end recovery times, plus the one-hour endpoint. Raw review timings remain unchanged. The logarithmic axis is in minutes; CSV times are in seconds. Final totals are computed, never fixed to historical counts. See the review README for scoring and timing caveats.
