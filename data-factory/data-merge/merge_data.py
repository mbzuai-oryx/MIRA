import os, json, ast, tqdm

full_info = json.loads(open('lmed_instruction_wrag.json').read())
write_list = []

for nowsetid in tqdm.tqdm(range(len(full_info))):
    if not os.path.exists(f"rewritten_data/{nowsetid}.json"):
        continue
    else:
        nowset = ast.literal_eval(full_info[nowsetid])
        rewritten_set = json.loads(open(f"rewritten_data/{nowsetid}.json").read())

        rewritten_set["conversations_original"] = nowset["conversations"]

        write_list.append(rewritten_set)

with open("lmed_instruction_rtra_51k.json", "w") as jsonf:
    json.dump(write_list, jsonf)

print("Finished!")