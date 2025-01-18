from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def market_standards_agent(llm, industry_name, company_name):
    prompt = PromptTemplate(
        input_variables=["industry_name", "company_name"],
        template=(
            "Analyze industry trends and standards in {industry_name} specific to AI, ML, and automation. "
            "Propose use cases for {company_name} to leverage GenAI, LLMs, and ML technologies to improve processes, "
            "enhance customer satisfaction, and boost operational efficiency."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(industry_name=industry_name, company_name=company_name)
