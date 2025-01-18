from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def resource_collection_agent(llm, use_cases):
    prompt = PromptTemplate(
        input_variables=["use_cases"],
        template=(
            "For the use cases: {use_cases}, find relevant datasets from platforms like Kaggle, HuggingFace, "
            "and GitHub. Save the links and recommend GenAI solutions such as document search, report generation, "
            "or AI-powered chat systems."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(use_cases=use_cases)
