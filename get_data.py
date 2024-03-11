import argparse
import pandas as pd
import os
import json
import requests

arg_parser = argparse.ArgumentParser(
                    prog='Data Downloader',
                    description='Download nessecary data from Github repos')
arg_parser.add_argument("--tsv_file", help="TSV file that contains links to Github repo and names.", required=True)
args = arg_parser.parse_args()

GH_API = "https://api.github.com/repos/{username}/{repository_name}/contents/"
OUT = "data"

def download_file(content_item_dict):
    download_url = content_item_dict["download_url"]
    file_req = requests.get(download_url)
    if file_req.status_code == 200:
        file_content = file_req.content.decode()
        return file_content
    else:
        raise Exception("Couldn't get file content.")

data = pd.read_csv(args.tsv_file, sep="\t")
data = data[~data["UD link"].isna()]

if not os.path.exists(OUT):
    os.mkdir(OUT)

for lang in data.iterrows():
    _, lang_data = lang
    lang_name = lang_data["language"]
    print(lang_name)
    # Create new dir where data will be stored.
    dir_name = "_".join(lang_name.strip().split())
    out_path = os.path.join(OUT, dir_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if len(os.listdir(out_path)) > 0:
        continue
    # Get content of Github repo.
    url = lang_data["UD link"]
    url = url.split("/")
    user, repo = url[-4], url[-3]
    gh_api = GH_API.format(username=user, repository_name=repo)
    r = requests.get(gh_api)
    content = json.loads(r.content.decode())
    for item in content:
        if item["name"].endswith(".conllu"): # Check for right file extension
            file_out = os.path.join(out_path, item["name"])
            file_content = download_file(item)
            print(file_out)
            with open(file_out, "w", encoding="utf-8") as file:
                file.write(file_content)
