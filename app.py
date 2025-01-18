import streamlit as st
from agent_modules.industry_research import industry_research_agent
from agent_modules.key_offerings import key_offerings_agent
from agent_modules.market_standards import market_standards_agent
from agent_modules.resource_collection import resource_collection_agent
from agent_modules.final_proposal import final_proposal_agent
from utils.tavily_tool import initialize_tavily_tool
from langchain.llms import HuggingFaceEndpoint
import os

# Set environment variables with actual API keys
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
tavily_api_key = os.getenv("TAVILY_API_KEY")


# Initialize LLM and Tavily tool
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7,token=huggingface_token)
tavily_tool = initialize_tavily_tool()

st.title("Multi-Agent Architecture for AI Use Cases")

company_name = st.text_input("Enter the Company Name:")
industry_name = st.text_input("Enter the Industry Name:")

if st.button("Run Agents"):
    with st.spinner("Running Industry Research..."):
        industry_data = industry_research_agent(llm, industry_name)
        st.subheader("Industry Research Results")
        st.write(industry_data)
    
    with st.spinner("Analyzing Key Offerings..."):
        key_offerings = key_offerings_agent(llm, company_name, industry_data)
        st.subheader("Key Offerings")
        st.write(key_offerings)
    
    with st.spinner("Generating Use Cases..."):
        use_cases = market_standards_agent(llm, industry_name, company_name)
        st.subheader("AI Use Cases")
        st.write(use_cases)
    
    with st.spinner("Collecting Resources..."):
        resources = resource_collection_agent(llm, use_cases)
        st.subheader("Resource Links")
        st.write(resources)
    
    with st.spinner("Compiling Final Proposal..."):
        proposal = final_proposal_agent(llm, use_cases, resources)
        st.subheader("Final Proposal")
        st.write(proposal)
