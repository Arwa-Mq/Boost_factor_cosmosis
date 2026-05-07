#!/bin/bash

data_path="/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles"
template="bf_all_bins.ini"
output_dir="/global/homes/a/arwa_mq/DESy3/Boost_factor_cosmosis/pipeline_all_bins/outputs_all_bins"

for l in 0 1 2 3; do
  for z in 0 1 2; do
    config="pipeline_l${l}_z${z}.ini"
    cp $template $config

    cat <<EOL >> $config

[boost_factor_likelihood]
file = /global/homes/a/arwa_mq/DESy3/Boost_factor_cosmosis/pipeline_all_bins/bf_all_like.py
data_path = $data_path
richness_bin = $l
redshift_bin = $z

[output]
filename = $output_dir/output_l${l}_z${z}.txt
format = text
EOL

    echo "Running Cosmosis for l=$l, z=$z..."
    cosmosis $config >& log_all&

  done
done

