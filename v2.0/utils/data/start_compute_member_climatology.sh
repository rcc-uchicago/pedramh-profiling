#!/bin/bash

data_dir=$1
#plev_units="Pa"
plev_units="hPa"
#sim_start=$2
#sim_end=$3
sims_in=$2
dependency_str=$3

#for (( sim=sim_start; sim<=sim_end; sim++ )); do
for sim in $sims_in; do
    #SIM=$sim
    #SIM2=$(( SIM + 22 ))
    #sims="${SIM},${SIM2}"
    sims=$sim
    #mean_days_list="3,5,7"
    #variable="ta,ua,va,hus,zg,tas,pl,pr_6h"
    variable="evap,hus,mrro,rmso,pl,pr_6h,snd,sndc,snm,ta,tas,ts,ua,va,zg"
    echo "Submitting sim ${sim} ${variable}"
    if [ -z "$dependency_str" ]; then
            sbatch --time=12:00:00 compute_member_climatology_short.py $data_dir $variable $plev_units $sims #$mean_days_list
    else
        sbatch --time=12:00:00 --dependency=afterany:$dependency_str compute_member_climatology_short.py $data_dir $variable $plev_units $sims #$mean_days_list
    fi
done
