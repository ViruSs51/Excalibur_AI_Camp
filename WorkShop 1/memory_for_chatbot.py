from groq import Groq
import groq
import re


class ChatFullMemory:
    __memory = {}

    def __init__(self, token: str):
        self.__client = Groq(api_key=token)

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

        answer_with_context = self.__getDefaultAnswer(prompt=f'<<<<{context}>>>>\n\n\nIn baza la informatia de mai sus(daca ea este. Daca nu, nu atrage atentia la tot ce eu am cerut sa faci. Doar raspunde la prompt de mai jos dupa doua puncte in limba in care el a fost scris) care este intre <<<<...>>>>, da-mi raspuns la text-ul/propmpt-ul de mai jos dupa doua puncte, in limba in care el este scris mai jos:\n{data}')
        
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

        return answer
    
    def summaryText(self, text) -> str:
        '''Summarizes the given text as concisely as possible while retaining the full context.
    
            Parameters:
            - text (str): The input text to be summarized.

            Returns:
            - str: A concise summary of the input text, maintaining the original context.'''
        
        summary_text = self.__getDefaultAnswer(prompt=f'Rezumeaza tot textul ce urmeaza dupa doua puncte mai jos in asa mod ca sa pastreze tot contextul dar sa fie rezumatul cat mai mic posibil(Daca nu este nimic de rezumat mai jos dupa doua puncte, nu raspunde nimic!):\n\n{text}')[1]

        return summary_text

if __name__ == '__main__':     
    chat = ChatFullMemory(token="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")
    
    while True:
        prompt = input('Input your ask: ')
        answer = chat.getAnswer(prompt=prompt)
        print(f'\n\n{answer}\n\n')