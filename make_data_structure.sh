#!/bin/bash

source keys.txt
afdc_key=$afdc_key


mkdir -p Data/Generated_Data
mkdir -p Data/AFDC
mkdir -p Data/State

if [ ! -f Data/AFDC/evse_stations.json ]; then
	afdc_url="https://developer.nrel.gov/api/alt-fuel-stations/v1.json?fuel_type=ELEC&limit=all&api_key=${afdc_key}"
	echo $afdc_url
	curl -o Data/AFDC/evse_stations.json $afdc_url
	echo "AFDC Data Downloaded"
else
	echo "AFDC Data Downloaded"
fi

if [ ! -f Data/State/tl_2023_us_state.zip ]; then
	url="https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
	curl -o Data/State/tl_2023_us_state.zip $url
else
	echo "State Geometries Downloaded"
fi

if [ ! -f Data/State/tl_2023_us_state.shp ]; then
	unzip Data/State/tl_2023_us_state.zip -d Data/State
	echo "State Geometries Unzipped"
else
	echo "State Geometries Unzipped"
fi

