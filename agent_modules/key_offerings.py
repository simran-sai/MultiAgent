from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def key_offerings_agent(llm, company_name, industry_data):
    prompt = PromptTemplate(
        input_variables=["company_name", "industry_data"],
        template=(
            "Using the findings about {company_name} in {industry_data}, identify their key offerings "
            "and strategic focus areas, and analyze their vision."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(company_name=company_name, industry_data=industry_data)
