#!/bin/bash
SCRIPT="record_optimized6.2.py"
LOGFILE="acquisition_log.txt"

echo "Press ESC to stop and remove all .npy files."

while true
do
    # Check if ESC is pressed
    read -t 0.1 -n 1 key
    if [[ $key == $'\e' ]]; then
        echo "ESC pressed, cleaning up..."
        rm *.npy
        exit 0
    fi

    echo "============================" | tee -a "$LOGFILE"
    echo "Starting acquisition at $(date)" | tee -a "$LOGFILE"

    PYTHONUNBUFFERED=1 python3 -u "$SCRIPT" 2>&1 \
      | awk '{ "date +\"%Y-%m-%d %H:%M:%S\"" | getline t; close("date +\"%Y-%m-%d %H:%M:%S\""); print "[" t "] " $0; fflush(); }' \
      | tee -a "$LOGFILE"

    echo "Acquisition stopped at $(date)" | tee -a "$LOGFILE"
    echo "Restarting in 2 seconds..." | tee -a "$LOGFILE"
    sleep 2
done
