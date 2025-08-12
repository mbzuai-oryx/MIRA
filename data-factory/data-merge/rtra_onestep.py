import ast
import os
import json, math
import base64
import io
from PIL import Image
from copy import deepcopy
from prompts_multimodal import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

rearrange_rag = rearrange()
rearrange_orag = rearrange_onlinerag()
initial_gen = initial_answer_generation()
rethinker = rethink()
final_gen = final_output_gen()

def main(current_conversation, rag_seq, rag_online, images=None):
    # Pass images to all multimodal processing functions
    rearrange_result = rearrange_rag.make_query(
        chat_history=current_conversation[:-1], 
        rag_info=rag_seq, 
        images=images
    )
    rearrange_orag_result = rearrange_orag.make_query(
        chat_history=current_conversation[:-1], 
        onlinerag=rag_online, 
        images=images
    )

    if rearrange_orag_result.startswith("<None>"):
        concanated_rag = rearrange_result
    else:
        concanated_rag = rearrange_result + "\nSearch API result: " + rearrange_orag_result
    
    initial_answer = initial_gen.make_query(
        rag_rewrite=concanated_rag, 
        chat_history=current_conversation[:-1], 
        images=images
    )

    rethink_result = rethinker.make_query(
        chat_history=current_conversation[:-1], 
        rag_rewrite=concanated_rag, 
        answer=initial_answer, 
        images=images
    )

    final_output = final_gen.make_query(
        chat_history=current_conversation[:-1], 
        rag_rewrite=concanated_rag, 
        answer=initial_answer, 
        thinking=rethink_result, 
        GT_answer=current_conversation[-1]["value"], 
        images=images
    )

    modified_gpt_reply = f"""Okay. I will now make use of the RAG data and initial conversation to generate a response. First I will find what's useful in RAG data and write them: <rearrange>{concanated_rag}</rearrange> From these data and the conversation, I think I can generate an initial answer as: <initial>{initial_answer}</initial> Let me think how to make it better: <rethink>{rethink_result}</rethink> After all, let's summarize everything, the best reply to the user will be <final>{final_output}</final>"""

    return modified_gpt_reply

def solve(nowsetid, nowset):
    # try:
        nowset_tosave = deepcopy(nowset)

        # Conversations
        nowcov = nowset["conversations"]
        stairs_conversation = []
        temp = []
        for i in range(len(nowcov)):
            temp.append(nowcov[i])
            if nowcov[i]["from"] == "gpt":
                stairs_conversation.append({"conv": deepcopy(temp), "idx": i})

        nowimage = nowset["image"]
        # Complete the Image.open() call and collect all images
        all_images = []
        
        # Process main image
        if nowimage:
            # Try different possible paths for the main image
            possible_main_paths = [
                nowimage,
                os.path.join("it_imfiles", nowimage),
                os.path.join("../../../清洗MRAG/it_imfiles", nowimage),
                os.path.join("rag_imfiles", nowimage)
            ]
            
            main_image_path = None
            for possible_path in possible_main_paths:
                if os.path.exists(possible_path):
                    main_image_path = possible_path
                    break
            
            if main_image_path:
                main_image_b64 = encode_image_to_base64(main_image_path)
                if main_image_b64:
                    all_images.append(main_image_b64)
        
        # RAGs
        nowrag = nowset["rag"]
        nowrag_ims = nowset["rag_img"]
        nowrag_online = nowset["rag_online"]
        nowrag_online_ims = nowset["rag_online_im"]

        # Process RAG images
        for rag_img in nowrag_ims:
            if "Image_File" in rag_img and rag_img["Image_File"]:
                # Construct full path - assume images are in relative directory
                img_path = rag_img["Image_File"]
                if not os.path.isabs(img_path):
                    # Try different possible base directories
                    possible_paths = [
                        img_path,
                        os.path.join("it_imfiles", os.path.basename(img_path)),
                        os.path.join("../../../清洗MRAG/it_imfiles", os.path.basename(img_path))
                    ]
                    for possible_path in possible_paths:
                        if os.path.exists(possible_path):
                            img_path = possible_path
                            break
                
                if os.path.exists(img_path):
                    img_b64 = encode_image_to_base64(img_path)
                    if img_b64:
                        all_images.append(img_b64)

        # Process online RAG images  
        # Note: These are URLs, would need to download first. For now, skip URL-based images
        # as they require network access and proper handling
        for online_img in nowrag_online_ims:
            # Could implement URL download logic here if needed
            # For now, we'll skip online images to avoid network dependencies
            pass

        rag_sequence = ""
        for m in nowrag:
            if "certainly!" in m["Content"].lower():
                continue
            rag_sequence+= m["Paragraph"] + ": " + m["Content"] + "\n"

        nowidx = 2
        for m in nowrag_ims:
            # Check if the image description is NaN (not a number)
            if not isinstance(m["Description"], str):
                # Skip this image or use a placeholder description
                m["Description"] = "(No description available)"

            rag_sequence+= "[Image " + str(nowidx) + "] which content is: " + m["Description"] + "\n"
            nowidx+=1
        for m in nowrag_online_ims:
            rag_sequence+= "[Image " + str(nowidx) + "] which content is: " + m["title"] + "\n"
            nowidx+=1

        nowset_tosave["rag_sequence"] = rag_sequence

        for now_stair_cov in stairs_conversation:
            # Pass all collected images to the main function
            rtra_gpt_reply = main(
                now_stair_cov["conv"], 
                rag_sequence, 
                nowrag_online[0], 
                images=all_images if all_images else None
            )
            nowset_tosave["conversations"][now_stair_cov["idx"]]["value"] = rtra_gpt_reply

        with open(f"rewritten_data/{nowsetid}.json", "w") as jsonf:
            json.dump(nowset_tosave, jsonf)
        
        print(f"Completed {nowsetid}")
    # except Exception as e:
    #     print(f"Error in {nowsetid}: {e}")
    
    # return True

if __name__ == '__main__':
    full_info = json.loads(open('lmed_instruction_wrag.json').read())
    fullinfo2 = []
    with open('rtra_instructions_raw_0203.json', "r") as f:
        for line in f:
            fullinfo2.append(json.loads(line))
    # full_info_2 = json.loads(open('rtra_instructions_raw_0203.json').read())
    tasklist = []

    # for nowsetid in range(len(fullinfo2)):
    for nowsetid in range(30):
        if os.path.exists(f"rewritten_data/{nowsetid}.json"):
            continue
        else:
            nowset = ast.literal_eval(full_info[nowsetid])
            nowset_2 = fullinfo2[nowsetid]
            nowset_2["rag_online"] = nowset["rag_online"]
            nowset_2["rag_online_im"] = nowset["rag_online_im"]
            tasklist.append((nowsetid, nowset_2))
        # if nowsetid == 43:
        #     solve(nowsetid, nowset_2)
    
    print("Starting workers, task sum: ", len(tasklist))
        
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(solve, task[0], task[1]) for task in tasklist]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                # You can add handling for the result if needed
            except Exception as e:
                print(f"Task generated an exception: {e}")
