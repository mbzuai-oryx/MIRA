import ast
import abc
import time
from openai import OpenAI

vl_model_base = "Qwen2-5-VL-72B-Instruct-128k"
client = OpenAI(
    api_key="sk-1",
    base_url="<Please input your VLLM Infra endpoint URL for QwenVL>"
)

def convert_listdic_to_string(conv):
    return_str = ""
    for m in conv:
        if "<image>\n" in m["value"]:
            return_str += m["from"] + ": " + m["value"].replace("<image>\n", "[Image 1] ") + "\n"
        else:
            return_str += m["from"] + ": " + m["value"] + "\n"
    return return_str

class prompting_base:
    def __init__(self, debug=False):
        self.server_name = None
        self.temperature = None
        self.debug = debug

    @abc.abstractmethod
    def prompt_generation(self, images: list = None, **kwargs):
        """
        Generates the prompt in OpenAI messages format.

        Args:
            images (list, optional): A list of base64 encoded image strings. Defaults to None.
            **kwargs: Additional arguments specific to the subclass implementation.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        pass

    def make_query(self, **kwargs):
        # starttime = time.time()

        # Pass images if present in kwargs
        images = kwargs.pop('images', None) 
        
        # Debug: print image count
        if self.debug:
            if images:
                print(f"Processing with {len(images)} images")
            else:
                print("Processing text-only (no images)")
            
        message = self.prompt_generation(images=images, **kwargs)
        
        # Check if message is already in the correct format (list of dicts)
        # If not, wrap it (though prompt_generation should now always return the correct format)
        if not isinstance(message, list):
             message = [{"role": "user", "content": message}] # Fallback, though ideally not needed

        # Validate message format for multimodal content
        if self.debug and images and len(message) > 0 and isinstance(message[0].get("content"), list):
            # Count text and image components
            content = message[0]["content"]
            text_count = sum(1 for item in content if item.get("type") == "text")
            image_count = sum(1 for item in content if item.get("type") == "image_url")
            print(f"Message contains {text_count} text components and {image_count} image components")

        try:
            response = client.chat.completions.create(
                messages=message,
                model=self.server_name,
                temperature=self.temperature,
                max_tokens=16*1024,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {e}")
            if self.debug:
                print(f"Message format: {type(message)}")
                if message:
                    print(f"First message keys: {message[0].keys() if message else 'Empty'}")
            # Fallback response
            return "Error occurred during processing."
        
        # print("Time used: ", time.time()-starttime)

class rearrange(prompting_base):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.server_name = vl_model_base
        self.temperature = 0.2

    def prompt_generation(self, chat_history, rag_info, images: list = None):
        text_prompt = """You are an advanced AI trained to analyze multimodal inputs (images, text, RAG data) and conversations. The user will give his conversation with GPT, please truncate irrelevant imnformation in RAG_data, and reply with the truncated RAG data. Do not modify contents in RAG data, just truncate irrelevant and reply. At least keep 1 from the dictionary and 1 image.
        [Chat history]
        {chat_history}
        [RAG data to be truncated]
        {rag_information}
        
        Keep original format like <n>: <value>. The truncated RAG data is: """.format(chat_history=convert_listdic_to_string(chat_history), rag_information=rag_info)
        
        content = [{"type": "text", "text": text_prompt}]
        
        # Add images if provided and not empty
        if images and len(images) > 0:
            for img_b64 in images:
                if img_b64:  # Ensure image data is not None or empty
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}" 
                        }
                    })
                
        return [{"role": "user", "content": content}]

class rearrange_onlinerag(prompting_base):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.server_name = vl_model_base # "Qwen2-5-VL-72B-Instruct-for-kimiv"
        self.temperature = 0.2

    def prompt_generation(self, chat_history, onlinerag, images: list = None):
        text_prompt = "You are an advanced AI trained to analyze multimodal inputs (images, text, RAG data) and conversations. The user will give his conversation with GPT, and GPT has obtained external knowledge for it to help answering. Please rewrite the external knowledge to make it more helpful to answering the conversation, and please only rewrite the existing things in the external knowledge. If the external knowledge is completely not about medical data, only write <None>. [Chat history]\n{chat_history}\n[External knowledge for rewriting]\n{rag_information} \nThe rewritten knowledge is: ".format(chat_history=convert_listdic_to_string(chat_history), rag_information=onlinerag)

        content = [{"type": "text", "text": text_prompt}]
        
        # Add images if provided and not empty
        if images and len(images) > 0:
            for img_b64 in images:
                if img_b64:  # Ensure image data is not None or empty
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })

        return [{"role": "user", "content": content}]

class initial_answer_generation(prompting_base):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.server_name = vl_model_base
        self.temperature = 0.2

    def prompt_generation(self, rag_rewrite, chat_history, images: list = None):
        text_prompt = """You are GPT, with the external knowledge you'll try to generate an initial answer to the medical question posed in the conversation. Please only output the content of your answer. Do not include any format. State it's from your database if you are using the external knowledge.
        Here are the retrieved information to help you think your answer: 
        [Your reference data]
        {rag_information} 
        [Chat history]
        {chat_history}
        Now I will answer the user with: """.format(rag_information=rag_rewrite, chat_history=convert_listdic_to_string(chat_history))

        content = [{"type": "text", "text": text_prompt}]
        
        # Add images if provided and not empty
        if images and len(images) > 0:
            for img_b64 in images:
                if img_b64:  # Ensure image data is not None or empty
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                
        return [{"role": "user", "content": content}]

class rethink(prompting_base):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.server_name = vl_model_base
        self.temperature = 0.2

    def prompt_generation(self, chat_history, rag_rewrite, answer, images: list = None):
        text_prompt = """You are a professional doctor and natural language scientist, and your fellow gpt has generated an initial answer to the medical question posed in the conversation. Now your task is to only answer the newest question in the [Chat history]. If you'll get up and refine the result for your fellow, what'll you think? Here are the information to help you think your answer: 
        [Chat history]
        {chat_history}
        [Previous Answer] 
        {answer}
        [Retrieved Knowledge]
        {rag_information}
        
        No need to write final answer, just directly write your plan about how you'kk refine it in 1 paragraph, marking with 1, 2, 3.""".format(rag_information=rag_rewrite, chat_history=convert_listdic_to_string(chat_history), answer=answer)

        content = [{"type": "text", "text": text_prompt}]
        
        # Add images if provided and not empty
        if images and len(images) > 0:
            for img_b64 in images:
                if img_b64:  # Ensure image data is not None or empty
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                
        return [{"role": "user", "content": content}]
    
class final_output_gen(prompting_base):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.server_name = vl_model_base
        self.temperature = 0.2

    def prompt_generation(self, chat_history, rag_rewrite, answer, thinking, GT_answer, images: list = None):
        text_prompt = """You are a professional doctor and natural language scientist, given the initial answer written by your fellow A, and you have thought how to refine it to better answer the user. Given the reference GT answer to the question, and you want your final answer have better use of the rag data than the GT answer, what will you refine your final answer to the user? Here are the information to help you think your answer:
        [Chat history]
        {chat_history}
        [Previous Answer]
        {answer}
        [Retrieved Knowledge]
        {rag_information}
        [Your thinking]
        {thinking}
        [GT answer]
        {GT_answer}

        Don't make it too significantly longer than or have diversed meaning with [GT answer]. The final answer you provide is: """.format(rag_information=rag_rewrite, chat_history=convert_listdic_to_string(chat_history), answer=answer, thinking=thinking, GT_answer=GT_answer)

        content = [{"type": "text", "text": text_prompt}]
        
        # Add images if provided and not empty
        if images and len(images) > 0:
            for img_b64 in images:
                if img_b64:  # Ensure image data is not None or empty
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                
        return [{"role": "user", "content": content}]