import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_chat_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature = 0
    )
    return model


def build_prompt():
    return ChatPromptTemplate.from_template(
        "You are an expert technical interviewer preparing questions for a candidate "
        "applying to the role described below.\n\n"
        "Job description content:\n{context}\n\n"
        "Task:\n"
        "1. Identify {num_categories} distinct core competency areas this role actually "
        "requires, based only on the content above. Do not invent tools, skills, or "
        "technologies that aren't mentioned or clearly implied in the text.\n"
        "2. For each category, generate exactly {questions_per_category} interview questions.\n"
        "3. Within each category, vary the style: include at least one conceptual question, "
        "one applied/scenario-based question, and one about a real technical trade-off.\n\n"
        "Format your response as:\n"
        "### [Category Name]\n"
        "1. Question text\n"
        "2. Question text\n"
        "...\n\n"
        "Output only the categorized questions — no preamble, no closing summary."
    )

def generate_interview_questions(chat_model, prompt_template, context:str, num_categories:int=5, questions_per_category:int=5):
    chain = prompt_template | chat_model
    response = chain.invoke({
        "context": context,
        "num_categories": num_categories,
        "questions_per_category": questions_per_category
    })
    return response.text