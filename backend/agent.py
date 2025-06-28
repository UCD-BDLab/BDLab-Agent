import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .scraper import scrape_members, scrape_research

class BDLabAgent:
    def __init__(self):
        # scrape fresh members and research
        members = scrape_members()
        research = scrape_research()
        history = []

        # history exists, load it
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as memory_file:
                memory = json.load(memory_file)
                history = memory.get("history", [])
        
        self.memory = {
            "members": members,
            "research": research,
            "history": history
        }

        # "cuda" if torch.cuda.is_available() else 
        self.device_nm = "cpu"
        # tokenizer and model initialization
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=True)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=True)
        self.model = self.model.to(self.device_nm)
        
    def save_memory(self):
        # save the current memory state to the json file
        with open("memory.json", "w") as f:
            json.dump(self.memory, f, indent=2)

    def build_prompt(self, user_query):
        # contextual prompt for the LLM
        # this method makes a big prompt string that tells the model how to answe based on 
        # what is the BDLab agent
        # who are the members
        # what research topics there are (soon) 
        # what has been said so far in the convo
        # and what is the user asking

        prompt = ""
        prompt += "You are BDLab-Agent, an assistant for CU Denver's Big Data Lab.\n"
        prompt += "Here are the lab members:\n"
        for member in self.memory["members"]:
            # list name and role only for brevity
            prompt += f"- {member.get('name', 'Unknown')} ({member.get('role', 'Unknown')})\n"

        prompt += "\nHere are the research topics:\n"
        for topic in self.memory["research"]:
            prompt += f"- {topic}\n"

        prompt += "\nConversation history:\n"
        for exchange in self.memory["history"]:
            prompt += f"User: {exchange.get('user', '')}\n"
            prompt += f"Agent: {exchange.get('agent', '')}\n"

        prompt += "\nUser asks: " + user_query + "\n"
        prompt += "Agent, please answer concisely based on the information above."
        return prompt

    def ask_llm(self, prompt):
        # when a user asks a questgions we first tokenize then try to generate a appropriate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_nm)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def handle_query(self, user_query):
        # query can be seen as an entry point for the agent to process prompts and return answers
        prompt = self.build_prompt(user_query)
        answer = self.ask_llm(prompt)

        # we take record of the conversation in memory and save it
        record = {"user": user_query,"agent": answer}
        self.memory["history"].append(record)
        self.save_memory()

        # return answser back to user.
        return answer

if __name__ == "__main__":
    # simple read, evaluate, print and loop. aka REPL
    agent = BDLabAgent()
    print("Welcome to BDLab-Agent! Type 'quit' to exit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = agent.handle_query(user_input)
        print("Agent:", response)
