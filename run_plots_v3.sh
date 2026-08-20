#!/bin/bash
set -e

PYTHON=/build_bak/mtk53686/semantic-entropy-probes/.venv/bin/python
export PYTHONPATH=/build_bak/UQ/UQ-transfer/src:/build_bak/UQ/python_packages

PAIRS_DIR=sep_scratch/transfer/hyperparam_sweep_v3
PAIR_NAMES="llama2_to_mistral llama32-1b_to_llama31-8b llama31-8b_to_qwen3-8b llama31-8b_to_phi4 llama31-8b_to_gemma llama31-8b_to_nemo"
PAIR_LABELS="Llama2-7B->Mistral-7B Llama3.2-1B->Llama3.1-8B Llama3.1-8B->Qwen3-8B Llama3.1-8B->Phi-4 Llama3.1-8B->Gemma-12B Llama3.1-8B->Mistral-Nemo"

cd /build_bak/UQ/UQ-transfer

for metric in auroc error_rate; do
    for axis in probe_budget map_budget; do
        echo "=== hyperparam sweep: $axis / $metric ==="
        $PYTHON -m sep.transfer.plot_hyperparam_sweep \
            --pairs-dir $PAIRS_DIR \
            --pair-names $PAIR_NAMES \
            --pair-labels $PAIR_LABELS \
            --token slt \
            --ridge-alphas 1e1 1e2 1e3 1e4 1e5 \
            --e2map-lams 0.1 1.0 10.0 100.0 1000.0 \
            --e2r0-lams 10 100 1000 10000 100000 \
            --e2rstar-lams 10 100 1000 10000 100000 \
            --metric $metric --axis $axis \
            --out-dir $PAIRS_DIR

        echo "=== best hyperparams: $axis / $metric ==="
        $PYTHON -m sep.transfer.plot_best_hyperparams \
            --pairs-dir $PAIRS_DIR \
            --e2map-lams 0.1 1.0 10.0 100.0 1000.0 \
            --e2r0-lams 10 100 1000 10000 100000 \
            --e2rstar-lams 10 100 1000 10000 100000 \
            --metric $metric --axis $axis
    done
done

echo ""
echo "=== All 8 plots written to $PAIRS_DIR ==="
echo ""
echo "Hyperparam sweep (all lambda/alpha curves):"
echo "  ALL_pairs_hyperparam_sweep_slt.png                  (probe_budget / auroc)"
echo "  ALL_pairs_hyperparam_sweep_slt_error_rate.png       (probe_budget / error_rate)"
echo "  ALL_pairs_hyperparam_sweep_slt_mapbudget.png        (map_budget   / auroc)"
echo "  ALL_pairs_hyperparam_sweep_slt_mapbudget_error_rate.png (map_budget / error_rate)"
echo ""
echo "Best hyperparams (one curve per method, best lambda auto-selected):"
echo "  ALL_pairs_best_hyperparams_slt.png                  (probe_budget / auroc)"
echo "  ALL_pairs_best_hyperparams_slt_error_rate.png       (probe_budget / error_rate)"
echo "  ALL_pairs_best_hyperparams_slt_mapbudget.png        (map_budget   / auroc)"
echo "  ALL_pairs_best_hyperparams_slt_mapbudget_error_rate.png (map_budget / error_rate)"
