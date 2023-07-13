from peft import PeftModel, PeftConfig
from transformers import  AutoModelForCausalLM, AutoTokenizer


class FalconConversation:
    def __init__(self, prompt="### Human: "):
        
        peft_model_id = "insomeniaT/falcon-7b-uae-qapairs-67"
        instruct_model_id = "tiiuae/falcon-7b-instruct"
        self.config = PeftConfig.from_pretrained(peft_model_id)
        self.sft_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path, trust_remote_code=True, device_map="auto"), peft_model_id, device_map="auto")
        self.sft_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", device_map="auto")
        self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path, trust_remote_code=True, device_map="auto")
        self.instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_id, trust_remote_code=True, device_map="auto")
        self.instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_id)
        
        self.models = {
            "base": self.base_model,
            "sft": self.sft_model,
            "instruct": self.instruct_model
        }
        
        self.tokenizers = {
            "base": self.sft_tokenizer,
            "sft": self.sft_tokenizer,
            "instruct": self.instruct_tokenizer
        }
        
        self.prompt = prompt

    def ask(self, question, model_name):
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        try:
            self.message = self.prompt + question + "### Assistant: "
            print(self.message)
            inputs = tokenizer(self.message, return_tensors="pt")
            outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=400, pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(result)
        except Exception as e:
            print(e)
            return e

        return result