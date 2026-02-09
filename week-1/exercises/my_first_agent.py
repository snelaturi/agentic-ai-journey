from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def chat():
    conversations = []

    print("Chatbot started.. Type exit to exit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Add user messages to conversation
        conversations.append({"role": "user", "content": user_input})
        conversations.append({"role": "system", "content": "You are a concise assistant. Reply in 1–2 short sentences only."})
        print("conversation call to model: ", conversations)
        #Get response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations,
            temperature=0.5, # 0 = deterministic, 1 = creative
            max_tokens=100
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response to conversation
        conversations.append({"role": "assistant", "content": assistant_message})

        print(f"Bot: {assistant_message} \n")
        print(f"Tokens used: {response.usage.total_tokens}")

if __name__ == "__main__":
    chat()



