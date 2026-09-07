import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class InterviewQuestion(BaseModel):
    category: str = Field(description="The competency area this question belongs to")
    question: str = Field(description="The full interview question text")

class InterviewQuestionSet(BaseModel):
    questions: List[InterviewQuestion]

class InterviewAnswer(BaseModel):
    index: int = Field(description="The number of the question this answers, matching the input list order")
    answer: str = Field(description="Sample answer to that question")

class InterviewAnswerSet(BaseModel):
    answers: List[InterviewAnswer]

def get_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        max_retries=5,
    )

def build_question_prompt():
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
        "one applied/scenario-based question, and one about a real technical trade-off."
    )

def generate_interview_questions(chat_model, prompt_template, context: str, num_categories: int = 5, questions_per_category: int = 5):
    structured_model = chat_model.with_structured_output(InterviewQuestionSet)
    chain = prompt_template | structured_model
    response = chain.invoke({
        "context": context,
        "num_categories": num_categories,
        "questions_per_category": questions_per_category
    })
    return response.questions  # 1 API call -> all questions

def build_answer_prompt():
    return ChatPromptTemplate.from_template(
        "You are helping a candidate prepare for an interview. For EACH numbered "
        "question below, write a strong sample answer using only the context "
        "provided for that question. Do not invent specifics about the candidate's "
        "own experience — describe what a strong answer should demonstrate.\n\n"
        "{qa_blocks}\n\n"
        "Return one answer per question, using the same index number given."
    )

def generate_all_answers(chat_model, prompt_template, qa_pairs: list):
    """qa_pairs: list of (question_text, context_text) tuples, in order"""
    structured_model = chat_model.with_structured_output(InterviewAnswerSet)
    chain = prompt_template | structured_model
    qa_blocks = "\n\n".join(
        f"[{i}] Context: {ctx}\n[{i}] Question: {q}"
        for i, (q, ctx) in enumerate(qa_pairs)
    )
    response = chain.invoke({"qa_blocks": qa_blocks})  # 1 API call -> all answers
    return response.answers