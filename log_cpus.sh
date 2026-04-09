#!/bin/bash
export PATH=/usr/local/slurm/current/bin:/usr/bin:/usr/local/bin:$PATH
USER=$(whoami)

mine=$(squeue -u $USER -t running -o "%C" -h | paste -sd+ | bc)
info=$(sinfo -o "%C" -h)
total=$(echo "$info" | cut -d/ -f4)
idle=$(echo "$info" | cut -d/ -f2)
echo "$(date '+%a %-m/%-d/%y %H:%M'): Using ${mine:-0} out of ${total}; ${idle} idle" >> ~/cpu_log.txt
# also print to stdout
echo "$(date '+%a %-m/%-d/%y %H:%M'): Using ${mine:-0} out of ${total}; ${idle} idle"
