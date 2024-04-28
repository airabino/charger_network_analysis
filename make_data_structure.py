import os
import requests
import json
import zipfile
import io

key = 'HGPBj8jd5JT96ixLhRl8wP970Ux3WHDZbye7EIrr'
path = 'Data/AFCD/'
file = 'evse_stations.json'

url = (
	f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?" + 
	f"fuel_type=ELEC&limit=all&state=CA&api_key={key}"
)

try:

	os.makedirs(path)

except OSError as e:

	pass

if not os.path.isfile(path + file):

	response = requests.get(url)

	with open(path + file, 'a') as file:

		json.dump(response._content.decode("utf-8"), file, indent = 4)


url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
path = 'Data/State/'
file = 'tl_2023_us_state.shp'

if not os.path.isfile(path + file):

	try:

		os.makedirs(path)

	except OSError as e:

		pass

	response = requests.get(url)
	zipped_data = zipfile.ZipFile(io.BytesIO(response._content))

	zipped_data.extractall(path)