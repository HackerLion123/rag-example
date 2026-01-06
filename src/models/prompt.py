from langchain_core.prompts import PromptTemplate


def create_ragqa_prompt():
    prompt = """You are an expect in answering questions based on provided context.
    Use the following context to answer the question.
    If you don't know the answer, say that you don't know.
    Make sure your answer is relevant to the question and is based only on the provided context.
    Keep the answer concise with not more than 350 words.
    Directly answer without adding extra text. 
    
    Question: {question}
    Context: {context}
    Answer:"""
    return PromptTemplate(template=prompt, input_variables=["question", "context"])


def create_grading_prompt():
    prompt = """You are a STRICT, DETERMINISTIC grader.

    You must output ONLY valid JSON with EXACTLY the keys shown below.
    All score values MUST be integers 0 or 1 (no decimals).
    The "reason" MUST be exactly ONE WORD from this set:
    OK INCORRECT INCOMPLETE OFFTOPIC UNCLEAR UNSUPPORTED

    Question: {question}
    Expected Answer: {expected_answer}
    Generated Answer: {generated_answer}

    Cross Encoder Evidence (higher means more relevant support):
    - answer_grade_score: {answer_grade_score}

    Rubric (binary):
    - correctness: 1 if answer is materially correct and based on given context, else 0
    - relevance: 1 if the answer is focused on the answering the question, else 0
    - clarity: 1 if the answer is understandable and well structured, else 0

    overall: 1 ONLY if correctness=1 AND relevance=1 AND ( expected_answer is vague and relevance is 1).
    passed: 1 ONLY if overall=1 AND answer_grade_score is not clearly contradictory to the answer.

    Respond ONLY with valid JSON (no extra text):
    {{"correctness":0, "relevance":0, "clarity":0, "overall":0, "reason":"INCORRECT/UNCLEAR"}}"""
    return PromptTemplate(
        template=prompt, 
        input_variables=[
            "question",
            "expected_answer",
            "generated_answer",
            "answer_grade_score"
        ]
    )


def create_citation_eval_prompt():
    prompt = """You are a STRICT, DETERMINISTIC citation checker.

    You must output ONLY valid JSON with EXACTLY the keys shown below.
    All values MUST be integers 0 or 1 (no decimals).
    The "reason" MUST be exactly ONE WORD from this set:
    OK MISSING IRRELEVANT UNSUPPORTED

    Answer: {answer}
    Citations: {citations}

    Rules (binary):
    - citation_relevance: 1 if at least one citation is topically relevant to the answer, else 0
    - citation_support: 1 if citations contain direct supporting information for the key claims, else 0

    Respond ONLY with valid JSON (no extra text):
    {{"citation_relevance":0, "citation_support":0, "reason":"MISSING}}"""
    return PromptTemplate(template=prompt, input_variables=["answer", "citations"])
