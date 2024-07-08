#!/bin/bash

# Start and end dates
start_date=20240102
end_date=20240430

while [ "$start_date" -le "$end_date" ]; do
    pipenv run python grid_predict.py $start_date
    start_date=$(date -d "$start_date + 1 day" +%Y%m%d)
done