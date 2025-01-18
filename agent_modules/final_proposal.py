from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def final_proposal_agent(llm, use_cases, resource_links):
    prompt = PromptTemplate(
        input_variables=["use_cases", "resource_links"],
        template=(
            "Compile the use cases: {use_cases}, ensuring relevance to the company’s goals and needs. "
            "Add references and resource links: {resource_links}. Include clickable links for the datasets."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(use_cases=use_cases, resource_links=resource_links)
