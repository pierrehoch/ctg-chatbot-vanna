import requests

base_url = "https://clinicaltrials.gov/api/v2/studies"
params = {
    "query.titles": "diabetes", # search keyword
    "pageSize": 5 # number of results per page
}

response = requests.get(base_url, params=params)

if response.status_code == 200:
    data = response.json()
    for study in data.get("studies", []):
        study_id = study["protocolSection"]["identificationModule"]
        print(f"NCT ID: {study_id['nctId']}, Title: {study_id['briefTitle']}")
else:
    print("Failed to retrieve data:", response.status_code)

# What are the products being developed in Asthma ? Give the product names, conditions, phases, company, name of clinical trials, expected end of the trial and statuses

# What are the last started trials ?