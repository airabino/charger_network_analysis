#!/bin/bash

source keys.txt
afdc_key=$afdc_key


mkdir -p Data/Generated_Data
mkdir -p Data/AFDC
mkdir -p Data/State
mkdir -p Data/Watts
mkdir -p Data/COUSUB

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

if [ ! -f Data/COUSUB/tl_2023_06_cousub.zip ]; then
	url="https://www2.census.gov/geo/tiger/TIGER2023/COUSUB/tl_2023_06_cousub.zip"
	curl -o Data/COUSUB/tl_2023_06_cousub.zip $url
else
	echo "COUSUB Geometries Downloaded"
fi

if [ ! -f Data/COUSUB/tl_2023_06_cousub.shp ]; then
	unzip Data/COUSUB/tl_2023_06_cousub.zip -d Data/COUSUB
	echo "COUSUB Geometries Unzipped"
else
	echo "COUSUB Geometries Unzipped"
fi

# Download files one by one
if [ ! -f Data/Watts/evwatts.public.connector.csv ]; then
curl -o "Data/Watts/evwatts.public.connector.csv" "https://livewire-data.s3.amazonaws.com/evwatts/evwatts.public/connector/evwatts.public.connector.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.connector.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=e2db42987cfeabea03b33bb3d95f8a07dfa4961e0da5a64ff008263a4ffb3e50"
fi

if [ ! -f Data/Watts/evwatts.public.dictionary.txt ]; then
curl -o "Data/Watts/evwatts.public.dictionary.txt" "https://livewire-data.s3.amazonaws.com/evwatts/evwatts.public/dictionary/evwatts.public.dictionary.txt?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.dictionary.txt%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=110786f9daad414cc8ca1d9d598a045648db0e445197dc0e6313c52dcf70dd86"
fi

if [ ! -f Data/Watts/evwatts.public.evse.csv ]; then
curl -o "Data/Watts/evwatts.public.evse.csv" "https://livewire-data.s3.amazonaws.com/evwatts/evwatts.public/evse/evwatts.public.evse.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.evse.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=d28dfdb0437aa97ee278486ade65fc8b6751648f05d3d5f5a4d19b1380220fa3"
fi

if [ ! -f Data/Watts/evwatts.public.session.csv ]; then
curl -o "Data/Watts/evwatts.public.session.csv" "https://livewire-data.s3.amazonaws.com/evwatts/evwatts.public/session/evwatts.public.session.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.session.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=be1f836e71e51cdd0ccca458e83a0cdb64cdbd9d09c592dc54ad0e2412e45d72"
fi

if [ ! -f Data/Watts/evwatts.public.vehicles.csv ]; then
curl -o "Data/Watts/evwatts.public.vehicles.csv" "https://livewire-ansible-prod-archive.s3.amazonaws.com/evwatts/evwatts.public/vehicles/evwatts.public.vehicles.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.vehicles.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=a88094ee25fec3a1a50b99077153edd34cad631ed125e042badba2637c6b59eb"
fi

if [ ! -f Data/Watts/evwatts.public.vehiclesessions.csv ]; then
curl -o "Data/Watts/evwatts.public.vehiclesessions.csv" "https://livewire-ansible-prod-archive.s3.amazonaws.com/evwatts/evwatts.public/vehiclesessions/evwatts.public.vehiclesessions.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.vehiclesessions.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=a1b5813c27787797faba733d4ac1bc5cc63c79de7b7eceed1364b8cb4488e88c"
fi

if [ ! -f Data/Watts/evwatts.public.vehicletrips.csv ]; then
curl -o "Data/Watts/evwatts.public.vehicletrips.csv" "https://livewire-ansible-prod-archive.s3.amazonaws.com/evwatts/evwatts.public/vehicletrips/evwatts.public.vehicletrips.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.vehicletrips.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GALUFNFR3%2F20240521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240521T204141Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHPkXZaoXr3MbCWKQsN1t8%2Bj4J9ZEXhzCTPn%2B6kys7klAiAsYRktYAIBr2ED9qiHIoN2Qa%2Fkb62Lz%2Bnvvr1POAUEGCr6AghOEAQaDDc0NzU2OTU3NTU2NCIM1EIiAb2HAm%2BcrbkaKtcCk0U0AVO0ZxLyEdMJKdmWWBR8VpXZgjOKuEzXll5bJaLwrUHIi3V7e2v8qzLoUJKp3Cwh04KAMGv1q%2BjfJ3fsIrUuptS5QZcxbeUWOWQNxLtrGsQyrNuGEQhnkqkAgPQeSSdT6LBwewIKCzKY4r3%2BRHRHgNbeaB4JU2X0lzrlwXwv%2Fp9U4FGXRn3ydaZiew4i8ZyoK2OOvtluhFdPaUya194ytgODnOFeCAYkqG4B8I7SQtIhT0q0%2BKP1xO1s7NFP1wsB25wE5%2FWan5tQ6qj5rPjQef8Wq3k2Lh6oBn6SOH104E28RdPPGRvqVZlDSIXpN6Q8rLEP0XbSl7%2F8Ocky%2F36KMcrMGQaSjYyE8M1UAM9y9k%2BB0b24pgLPTpj55%2F45gEjxkpSDDdDK5ewn7JNAEKU7UZ0cYcPKamqNjRflJUXxYiZ%2BTWfAY05PCJuOTDE7tJfmykApZzCDjrSyBjqfAQED3jVBqKsiLMilksaheleZMaOPcQS67bMnfftfik7db3bkefMMj8rth9mYvpbM0C3w%2BWCZYVGtvQxzlJxv2GF2Ysgh32tPJdpWWT%2FU7YVUiwRUmRH51aD6Cm9jnMwxMeTvS11DrjjJ%2FJS9UYwaiJhOKoXlslWKY5fZcMdcA%2FehGC01YLWVxym19h0f%2B6RWlPA2GpecdZvQSWQrtB6R9A%3D%3D&X-Amz-Signature=6b5e5535fcc83990ca57f911b09020000d35cd6a8328f4a1ec861cee8b6192e0"
fi