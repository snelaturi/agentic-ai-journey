from pyexpat.errors import messages

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI



load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Simple chat
# messages = [
#      SystemMessage("You are a helpful agent! give sharp answers in 1 or 2 lines"),
#      HumanMessage("What is an AI agent ?"),
# ]
#
# response = llm.invoke(messages)
#
# print(response.content)

# Chat with Memory

# memory = ConversationBufferMemory(return_messages=True, memory_key="history")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a personal assistant and give me answer in 1 or 2 lines"),
    # MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Attach memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input"
)

print("Chat started! Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == 'quit':
        break

    if not user_input:
        continue

    # Invoke chain
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user1"}}
    )

    print(f"\nBot: {response.content}\n")








