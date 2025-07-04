import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SiteAgent:
    def __init__(self, site_data):
        self.site_data = site_data
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=True, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)
        self.sections = {}

        self._build_people_section()
        self._build_research_section()
        self._build_education_section()
        self._build_news_section()

    def _extract_people_json(self):
        for page in self.site_data.values():
            je = page.get("json_equivalent", {})
            if "https://cse.ucdenver.edu/~bdlab/people.json" in je:
                return je["https://cse.ucdenver.edu/~bdlab/people.json"]

        return None

    def _extract_json_list(self, filename):
        for page in self.site_data.values():
            for url, data in page.get("json_equivalent", {}).items():

                if url.endswith(filename) and isinstance(data, list):
                    return data
                
        return None

    def _build_people_section(self):
        ppl = self._extract_people_json()

        if not ppl:
            return

        text = ""
        for group in ["professors", "team_members", "staff", "alumni"]:
            if group in ppl:
                title = group.replace("_", " ").title() + ":\n"
                text += title

                members = ppl[group]
                for rec in members.values():
                    line = "- " + rec["name"] + " (" + rec["role"] + ")\n"
                    text += line

                text += "\n"

        self.sections["people"] = text.strip()

    def _build_research_section(self):
        research = self._extract_json_list("research.json")

        if not research:
            return

        text = ""

        for item in research:
            line = "- " + item["title"] + ": " + item["summary"] + "\n"
            text += line

        self.sections["research"] = text.strip()

    def _build_education_section(self):
        education = self._extract_json_list("education.json")

        if not education:
            return

        text = ""
        for course in education:
            desc_list = course.get("description", [])
            desc = ""
            
            for d in desc_list:
                if desc:
                    desc += ", "
                desc += d

            line = "- " + course["name"] + ": " + desc + "\n"
            text += line

        self.sections["education"] = text.strip()

    def _build_news_section(self):
        news = self._extract_json_list("news.json")

        if not news:
            return

        text = ""
        for item in news:
            date_str = item["month"] + " " + str(item["date"]) + ", " + str(item["year"])
            line = "- " + date_str + ": " + item["title"] + "\n"
            text += line

        self.sections["news"] = text.strip()

    def _ask(self, prompt):
        tokens = self.tokenizer( prompt,return_tensors="pt").to(DEVICE)
        out = self.model.generate(**tokens, max_new_tokens=200)
        gen = out[0][tokens["input_ids"].shape[-1]:]

        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def query(self, user_q):
        low = user_q.lower()

        if ("people" in low or "member" in low or "staff" in low
                or "alumni" in low or "professor" in low):
            
            summary = self.sections.get("people")
            if not summary:
                return "Sorry, I couldn't find any people data."

            prompt = "You are BDLab-Agent. Here are the lab people:\n\n" + summary + "\n\nUser question: " + user_q + "\nAnswer using only the above information." 
            return self._ask(prompt)

        for key in ["research", "education", "news"]:
            if key in low:

                summary = self.sections.get(key)
                if not summary:
                    return "Sorry, I couldn't find any " + key + " data."

                prompt = "You are BDLab-Agent. Use ONLY this " + key + " info to answer:\n\n" + summary + "\n\nUser question: " + user_q + "\nAnswer concisely." 
                return self._ask(prompt)

        return "Sorry, I only know about people, research, education, and news."


if __name__ == "__main__":
    with open("simple_scrape.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    agent = SiteAgent(data)
    print("Ask me anything about the site! (Type 'quit' to exit.) or crtl+c to stop the program.")
    while True:
        que = input("You: ").strip()
        if que.lower() in ("quit", "exit"):
            break

        print("Agent:", agent.query(que))
