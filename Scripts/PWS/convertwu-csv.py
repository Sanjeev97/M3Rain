# Python program to convert
# JSON file to CSV

import pandas as pd
import json

with open('KLALAKEC144-20220101.json', 'r') as f:
    data = json.load(f)
df = pd.json_normalize(data)
df.to_csv('KLALAKEC144-20220101.csv')
