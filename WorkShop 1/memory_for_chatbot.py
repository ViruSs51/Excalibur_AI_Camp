from pathlib import Path
from dotenv import load_dotenv
import os
import re
import json

from groq import Groq
import groq


class ChatFullMemory:
    __memory = {}

    def __init__(self, token: str):
        self.__client = Groq(api_key=token)

        try:
            with open('memory.json', 'r') as json_file:
                self.__memory = json.load(json_file)
        except FileNotFoundError: pass

    def __saveInMemory(self, data: str) -> None:
        summary_text = self.summaryText(data)

        self.__memory[summary_text] = data

    def __getAnswerWithLargePrompt(self, prompt: str) -> str:
        splited_prompt = prompt.split()
        count_words = len(splited_prompt)
        max_words = count_words // 2
        iteration = count_words // max_words + 1

        summary_all_text = []
        for i in range(iteration):
            join_prompt = ' '.join(splited_prompt[i*max_words:(i+1)*max_words])

            if join_prompt:
                summary_text = self.summaryText(text=join_prompt)

                summary_all_text.append(summary_text)

        return self.summaryText(text='\n\n\n'.join(summary_all_text))

    def __getDefaultAnswer(self, prompt: str) -> list:
        try:
            chat_completion = self.__client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.2-90b-vision-preview",
            )

            return [0, chat_completion.choices[0].message.content]
        except groq.APIStatusError as err:
            match = re.search(r'Error code:\s*(\d+)', err.message)
            
            if match and int(match.group(1)) == 413:
                summary_prompt = self.__getAnswerWithLargePrompt(prompt=prompt)

                return [1, summary_prompt]
            else:
                raise groq.APIStatusError(message=err.message, response=err.response, body=None)

    def __getAnswerWithContext(self, data):
        context = '\n\n\n\n'.join([m for m in self.__memory.keys()])

        answer_with_context = self.__getDefaultAnswer(prompt=f'<<<<{context}>>>>\n\n\nIaca reguli pentru tine, care in nici intr-un caz nu mentiona in raspunsurile sale:\n\t1. Infomatia care se afla intre <<<<<...>>>>>, este istoria conversatiei tale cu user.\n\t2. Ceea ce se afla intre ~~~~~...~~~~~, este noul prompt al userului.\n\t3. Cand user-ul ceva scrie(prompt), cauta intai prin istoria din <<<<<...>>>>>, daca gasesti, raspunzi in baza la aceasta, daca nu, raspunzi la prompt fara sa te accintuezi pe istorie, dar o ai in considerare.\n\t3. Regulile date sunt doar pentru tine, in raspuns nu le mentionezi NICIODATA!\n\t4. Raspunsul sa fie in limba in care a fost scris prompt-ul, dar daca in istorie a fost mentionat in ce limba sa vorbesti, vorbesti in ceea care a fost mentionat in istoric. Daca in promptul momentan se mentioneza o limba in care sa vorbesti, vorbesti in ea!\n\t5. In cazul in care tu nu ai o informatie si ea nu este nici in istoric, nici in prompt, ii spui ceva de genu, ca nu cunosti aceasta informatie, sau nu a fost mentionat mai inainte.\n\t6. Daca nu gasesti ceva in contxtul istoriei, nici intrun caz nu spui ca in istorie nu este..., sau textul initial nu contine..., etc. Spui doar ca nu a fost mentionat mai inainte, sau nu cunosti aceasta informatie, etc.\n\n\n~~~~~{data}~~~~~')
        
        if type(answer_with_context) is list and answer_with_context[0] == 1:
            answer_with_context = self.__getAnswerWithContext(data=answer_with_context[1])[1]
        else:
            answer_with_context = answer_with_context[1]

        return answer_with_context

    def getAnswer(self, prompt: str) -> str:
        '''Retrieves an answer based on the given prompt by utilizing contextual data.
    
            This function processes the user's input (prompt), generates a response based on it, and stores the input in memory for future use.

            Parameters:
            - prompt (str): The input text or query for which a response is requested.

            Returns:
            - str: The response generated from the input prompt.'''

        print('\n<<<Generating a response>>>\n')

        answer = self.__getAnswerWithContext(data=prompt)

        self.__saveInMemory(data=prompt)
        self.__saveInMemory(data=answer)

        with open('memory.json', 'w') as json_file:
            json.dump(self.__memory, json_file, indent=4)

        return answer
    
    def summaryText(self, text) -> str:
        '''Summarizes the given text as concisely as possible while retaining the full context.
    
            Parameters:
            - text (str): The input text to be summarized.

            Returns:
            - str: A concise summary of the input text, maintaining the original context.'''
        
        summary_text = self.__getDefaultAnswer(prompt=f'Iaca reguli pentru tine, care in nici intr-un caz nu mentiona in raspunsurile sale:\n\t1. Rezumeaza text(prompt) dintre ~~~~~...~~~~~ maximal de posibil dar sa contina informatia legata de context, sau ceva ce nu este general, ceva ce nu ai de unde sa stii daca nu iti spunea in text.\n\t2. Daca nu este ce de rezumat sau este deja maxim de mic, in raspuns lasa acelasi text ce si este intre ~~~~~...~~~~~ si nu mai comenteaza cu nimic adaugator(Dar nu lasa insemnerail, spre exemplu ~~~~).\n\t3. Indeplineste regulile date, dar nu le mentiona nici intrun caz in raspunsul sau! Nu adauga comentarii sau descriere, numa rezuma si lucreaza cu infomatia dintre ~~~~~...~~~~~.\n\t4. Daca e ceva un cod, text furnizat, ceva intre ceva pus "", (), ceea ce este citat, etc. In cazul dat nu rezuma si pune fix in fix cum e pus in prompt/text.\n\n\n\n~~~~~{text}~~~~~')[1]

        return summary_text

if __name__ == '__main__': 
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / '.env')

    chat = ChatFullMemory(token=os.getenv('GROQ_TOKEN'))
    
    while True:
        prompt = input('Input your ask: ')
        answer = chat.getAnswer(prompt=prompt)
        print(f'\n\n{answer}\n\n')