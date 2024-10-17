import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig
import colorful as cf
from num2words import num2words
from retrieval_load import Retrieval

cf.use_true_colors()
cf.use_style('monokai')


class Agent:
    def __init__(self):
        print(cf.bold | cf.purple("LOAD"))
        checkpoint = "jihyoung/EMMA"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
        self.tokenizer.model_max_length = 1024
        self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(self.model,
                                               checkpoint,
                                               torch_dtype=torch.bfloat16,
                                               is_trainable=False)
        self.model = self.model.to(self.device)
        self.retrieval = Retrieval()
        self.conversation_history = []
        self.session_count = 1
        self.sequence = ""
        self.speaker = []
        self.memory = []

    def observe(self, observation):
        self.conversation_history.append(observation)

    def set_input(self):
        input_text = " ".join(self.conversation_history)
        input_text = "{}{}".format(self.sequence, input_text)
        return input_text

    def generate(self, user_response):
        self.observe(user_response)
        input_text = self.set_input()

        if self.session_count != 1:
            target_memory = self.retrieval.predict(input_text, self.memory)
            memory_sequence = f" [MEMORY] {target_memory} "
            link_memory = self.link(target_memory)
            if len(link_memory) != 0:
                for sentence in link_memory:
                    memory_sequence += f"[LINK] {sentence} "
                temp = input_text.split(' [NOW] ')
                input_text = temp[0] + memory_sequence + f" [NOW] {temp[1]}"
            else:
                temp = input_text.split(' [NOW] ')
                input_text = temp[0] + memory_sequence + f" [NOW] {temp[1]}"

        input_text = "generation: " + input_text
        print(input_text)

        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95,
                                      do_sample=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if response != "[END]":
            self.observe(response)

        return response

    def summary(self, prev_session):
        for idx in range(2):
            input_text = self.set_input()
            input_text = f"summarize [ABOUT_{self.speaker[idx][0]}]: " + input_text

            inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95,
                                          do_sample=True)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if '[NONE]' in response:
                continue
            elif '[SEP]' in response:
                response = response.split('[SEP]')
                for sentence in response:
                    sentence = str(sentence).strip()
                    sentence += f" (about {self.speaker[idx][0]}, from {prev_session} session)"
                    self.memory.append(sentence)
            else:
                response = str(response).strip()
                response += f" (about {self.speaker[idx][0]}, from {prev_session} session)"
                self.memory.append(response)

        print(self.memory)

    def link(self, source):
        link_memory = []

        for sentence in self.memory:
            if source == sentence:
                continue
            pair_check = f"memory sentence1: {source} memory sentence2: {sentence}"

            inputs = self.tokenizer([pair_check], return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95,
                                          do_sample=True)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if 'positive' in response:
                link_memory.append(sentence)

        return link_memory

    def reset_history(self):
        self.conversation_history = []

    def run(self):
        def get_valid_input(prompt, default):
            while True:
                user_input = input(prompt)
                if user_input in ["Y", "N", "y", "n"]:
                    return user_input
                if user_input == "":
                    return default

        model_name = input(cf.orange("Please write model's name: "))
        model_job = input(cf.orange("Please write model's job: "))

        self.session_count = 1
        while True:
            session_num = num2words(self.session_count, to='ordinal')
            if self.session_count > 1:
                self.summary(prev_session=num2words(self.session_count - 1, to='ordinal'))
                self.reset_history()

            user_name = input(cf.orange("Please write user's name: "))
            user_job = input(cf.orange("Please write user's job: "))
            self.speaker = [[model_name, model_job], [user_name, user_job]]

            prefix = f"[{self.speaker[0][0]}] {self.speaker[0][1]} [{self.speaker[1][0]}] {self.speaker[1][1]} "
            self.sequence = prefix + f"[NOW] {session_num} session "

            self.chat()
            self.session_count = self.session_count + 1
            continue_chat = get_valid_input(cf.purple("Start a new session with new event? [Y/N]: "), "Y")
            if continue_chat in ["N", "n"]:
                break

        print(cf.blue("END"))

    def chat(self):
        print(cf.green(
            "Chat with EMMA! Input [NEXT] to switch to the next session and [END] to end the conversation."))
        while True:
            user_input = input("You: ")
            if user_input == "[NEXT]" or user_input == "[END]":
                break
            user_input = "[{}] {} [{}]".format(self.speaker[1][0], user_input, self.speaker[0][0])
            response = self.generate(user_input)
            print(cf.blue("EMMA: " + response))


def main():
    print(cf.bold | cf.blue("START"))
    emma = Agent()
    emma.run()


if __name__ == '__main__':
    main()
