from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def industry_research_agent(llm, industry_name):
    prompt = PromptTemplate(
        input_variables=["industry_name"],
        template=(
            "Research the industry {industry_name}. Provide key trends, challenges, "
            "major players, and strategic focus areas such as operations, supply chain, and customer experience."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(industry_name=industry_name)
