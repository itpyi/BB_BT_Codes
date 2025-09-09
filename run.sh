#!/bin/bash
#
#
export QEC_DEBUG_DIAGRAM=1
# python simulation_generic.py --config config_examples/construct_bt_from_bb_small_with_metacheck.json --output-dir test_meta_check --multiprocess
python simulation_generic.py --config config_examples/construct_bt_from_bb_small_no_metacheck.json --output-dir test_no_meta_check --multiprocess


python std_monomial_basis_f2.py \
  --vars x y \
  --gens "y + y**2 + x**3" "x + x**2 + y**3" \
  --periods "x:12" "y:6" \
  --order lex \
  --nf "1 + x**7*y**2"