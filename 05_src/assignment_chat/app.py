
import os
import sys
from dotenv import load_dotenv

load_dotenv("../05_src/.env")
load_dotenv("../05_src/.secrets")

sys.path.append("../05_src")
import requests
from typing import List, Dict
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from .rag_pipeline import initialize_vectorstore
import sys
from dotenv import load_dotenv



# 1. Disable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"


from openai import OpenAI
from pydantic import BaseModel, Field
from utils.clients import get_client
import os
# os.environ["LANGSMITH_TRACING"] = "false"
print(os.getcwd())
from utils.logger import get_logger
from utils.clients import get_client
MODEL = os.getenv('MODEL', 'gpt-4o-mini')
_logs = get_logger(__name__)
client = get_client()

# Initialize local vectorstore retriever
retriever = initialize_vectorstore()

# 2. The prompt has been designed with specific instructions, to provide guidance within the context and boundaries of the guardrails

system_prompt = (
    "You are a strict technical assistant for an A-Z wall-mounting guide covering "
    "drywall, concrete, and other wall types.\n\n"

    "You will be asked to provide guidance on mounting various object types "
    "on various wall types.\n\n"

    "Your ONLY source of knowledge is the provided Context below.\n"
    "Do NOT use outside knowledge. Do NOT guess, assume, or invent instructions.\n\n"

    "Follow these rules IN ORDER:\n"

    "1. Identify the object type, wall substrate, object weight, and other relevant "
    "information provided by the user.\n\n"

    "2. NEVER ask the user what tools, hardware, anchors, fasteners, or equipment "
    "they have available. Use the Context to recommend the appropriate tools and "
    "hardware directly.\n\n"

    "3. If information such as object weight or another selection criterion is missing, "
    "do NOT ask the user for that information. Instead, provide the different applicable "
    "possibilities described in the Context. Clearly explain which recommendation applies "
    "to each condition. Include ONLY possibilities supported by the Context.\n\n"

    "4. If the user's wall substrate is specified, ONLY provide possibilities that "
    "apply to that wall substrate. Do not include recommendations for other wall types.\n\n"

    "5. ONLY use the refusal response when the Context does not contain enough information "
    "to provide any applicable recommendation. Respond ONLY with: "
    "'I am sorry, but the provided A-Z wall-mounting guide does not contain information "
    "on that topic.'\n\n"

    "6. When relevant information appears in multiple Context passages, combine those "
    "passages ONLY when they apply to the same object type and wall substrate.\n\n"

    "7. Match the object type, wall substrate, object weight, and other applicable "
    "criteria to the information provided in the Context before selecting recommendations.\n\n"

    "8. When the Context contains the information, include both the required tools "
    "and installation steps in your answer.\n\n"

    "9. When selecting hardware, anchors, fasteners, or tools, use ONLY the "
    "recommendations specifically listed for the applicable wall substrate. "
    "Do not combine hardware recommendations from different wall substrates.\n\n"

    "10. If the Context contains a table or decision matrix, use ONLY values from "
    "the column corresponding to the user's specified wall substrate. If multiple "
    "values apply because another criterion is unknown, explain each applicable "
    "possibility separately.\n\n"

    "11. Before answering, verify that every recommended anchor, fastener, tool, and "
    "installation step applies to the user's specified wall substrate according to "
    "the Context.\n\n"

    "12. NEVER write, generate, or include a 'Sources Consulted' section, "
    "source list, page list, or citation footer in your response. "
    "Stop your answer after the technical response. "
    "The application will add sources separately.\n\n"

    "Context:\n"
    "{context}"
)
    
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. This function makes an API call to Serpapi service with an item name as parameter and returns shopping links

def find_items(items: list[str]) -> list[str]:
    results = []

  

    for item in items:
       
        params={
                    "engine": "google_shopping",
                    "q": item,
                    "json_restrictor": "shopping_results[0:3].{title,product_link}",
                    "api_key": os.getenv("SERPAPI_KEY")
                }
        
        res = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        data = res.json()

        for product in data.get("shopping_results", []):
            link = product.get("product_link")
            print(product.get("title"), "->", link)

            if link:
                results.append(link)
        print("Status:", res.status_code)
        print("API key loaded:", bool(os.getenv("SERPAPI_KEY")))
        print("Keys returned:", data.keys())
        print(data)
        

    return results


item_list = []

# 4. Handles the user chat interaction and generates the assistant response

def simple_chat(message: str, history: List[Dict]) -> str:
    langchain_messages = []
    for msg in history:
        if msg.get('role') == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg.get('role') == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))


    retrieved_docs = retriever.invoke(message)
    
    # Format context string (MUST BE DEFINED HERE BEFORE CHAIN EXECUTION)
    context_str = format_docs(retrieved_docs)

    # 3. Extract page numbers from chunk metadata
    source_pages = []
    for doc in retrieved_docs:
        page = (
            doc.metadata.get('page') or 
            doc.metadata.get('page_label') or 
            doc.metadata.get('custom_id')
        )
        if page is not None and str(page) not in source_pages:
            source_pages.append(str(page))

    # 4. Execute LLM chain



    formatted_prompt = prompt.invoke({
    "context": context_str,
    "chat_history": langchain_messages,
    "input": message
    })



    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": (
                    "user" if msg.type == "human"
                    else "assistant" if msg.type == "ai"
                    else msg.type
                ),
                "content": msg.content
            }
            for msg in formatted_prompt.messages
        ],
        temperature=0
    )
    answer = response.choices[0].message.content

# Instruct the model to extract the item to be mounted from user's response and pass it on to find_items()

    if len(langchain_messages) == 0:
        product_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the object being hung from the user's message. "
                        "Return ONLY the object name as a single word. "
                        "No sentence, explanation, punctuation, weight, or wall type.\n\n"
                        + message
                    )
                
                }
            ],
            temperature=0
        )
        products = product_response.choices[0].message.content
        item_list.append(products)
        print("Identified Products:", products)
        product_links = find_items([products])

        if product_links:
            answer += f'\n\n---\n**Shopping for a product? Find inspirations for "{products}" below:**\n'
            for link in product_links:
                answer += f"- {link}\n"
    elif len(langchain_messages) >= 1:
            product_response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Extract the object being hung from the user's message. "
                                    "Return ONLY the object name as a single word. "
                                    "No sentence, explanation, punctuation, weight, or wall type.\n\n"
                                    + message
                                )
                            
                            }
                        ],
                        temperature=0
                )
            
            
            products = product_response.choices[0].message.content.strip()
            item_list.append(products)
            print("Identified Products:", products)
            if (item_list[-1].lower() != item_list[-2].lower()) and item_list[-1].lower() != 'weight':
                product_links = find_items([products])
                if product_links:
                        answer += f'\n\n---\n**Shopping for a product? Find inspirations for "{products}" below:**\n'
                        for link in product_links:
                            answer += f"- {link}\n"

    print(len(langchain_messages))
    print(item_list)
            
        


# 5. Append sources footer to the answer
    if source_pages:
        sources_footer = "\n\n---\n**Sources Consulted:** Page " + ", Page ".join(source_pages)
        answer += sources_footer
    else:
        answer += "\n\n---\n**Sources Consulted:** Drywall Guide (Metadata missing)"
    
  
    return answer
   


# 6. Customize Gradio's UI theme 

theme = gr.themes.Default(
    primary_hue="red",
    secondary_hue="pink",
    spacing_size="sm"
).set(
    body_background_fill_dark="#1e293b",
    block_background_fill_dark="#0f172a",
    background_fill_primary_dark="#1e293b",
    background_fill_secondary_dark="#0f172a",
    body_text_color_dark="#f8fafc"
)

js = """
function forceDarkMode() {
    const url = new URL(window.location.href);

    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.toString());
    }
}
"""

demo = gr.ChatInterface(
    fn=simple_chat,
    title="Wall Mounting RAG Assistant"
)

demo.launch(
    theme=theme,
    js=js
)

