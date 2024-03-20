# %%
import json

path = '/data/codebase/ireg/misc/ireg_data_collection/refcoco+_vlt5_ofa_scst_combine_clamp_mmi_refcoco+_train_bad_sents.json'

with open(path, 'r') as fn:
    d = json.load(fn)
    
print("Doing...")
for idx, itum in enumerate(d):
    itum['uni_id'] = idx
    
    
with open(path, 'w') as fn:
    json.dump(d, fn)
    
print("Finished!!!")

# %%
print("This is great!")

# %%
