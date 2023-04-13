from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, CombinedMemory
from langchain.memory import RedisChatMessageHistory

import torch

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0
}

model_name = "chavinlo/alpaca-native"

max_memory_mapping = {0: "5GB"}

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", offload_folder="offload", quantization_config=quantization_config, max_memory=max_memory_mapping)

pipe = pipeline('text-generation',
                model=model,
                tokenizer=tokenizer,
                max_length=256,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.2)

llm=HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes

### Instruction:
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt,
                     llm=llm)

# question = "What is the capital of India?"

# print(llm_chain.run(question))

# while True:
#     print("Type in anything you wish to ask Alpaca")
#     question = input("")
#     print(llm_chain.run(question))

redis = RedisChatMessageHistory("jj-alpaca")

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=redis
)

conversation.prompt.template = '''
Current conversation:
{history}
Human: {input}
AI:'''

conversation.predict(input="Hello! I am Jenish. Who are you?")