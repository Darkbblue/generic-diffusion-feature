# this script lets you edit many layers' config, alleviating manual labor
import json

file_in = 'configs/config_xl_full.json'
file_out = 'configs/config_xl_analysis.json'

with open(file_in, 'r') as f:
	file = json.load(f)

def edit(content, target):
	for entry in file.keys():
		for c in content:
			if c in entry:
				file[entry] = target
				break

edit(['down', 'mid'], False)

with open(file_out, 'w') as f:
	f.write(json.dumps(file))
