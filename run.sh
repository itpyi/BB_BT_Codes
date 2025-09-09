#!/bin/bash
#
#
export QEC_DEBUG_DIAGRAM=1
# python simulation_generic.py --config config_examples/construct_bt_from_bb_small_with_metacheck.json --output-dir test_meta_check --multiprocess
python simulation_generic.py --config config_examples/construct_bt_from_bb_small_no_metacheck.json --output-dir test_no_meta_check --multiprocess
