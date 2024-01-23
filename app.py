from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.chains import create_extraction_chain

find_news_articles = DuckDuckGoSearchResults(backend="news")
find_news = Tool.from_function(
    func=find_news_articles,
    name="ArticlesFinder",
    description="Find the latest news articles"
)

llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key = "OPENAI_API_KEY")
structured_schema = {
    "properties": {
        "title": {"type": "string"},
        "summary":{"type":"string"},
        "url": {"type":"string"}
    },
    "required": ["title", "url"],
}

extraction_chain = create_extraction_chain(structured_schema, llm)

final_result_tool = Tool.from_function(
    func=extraction_chain.run,
    name="Final result",
    description="Output the news articles"
)

tools = [find_news, final_result_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)


# Streamlit App
import streamlit as st 

def main():
    st.title("News Search Agent")

    query = st.text_input("Enter your news search query:")
    
    prompt = f"""You are a news search agent. Provide the latest news about '{query}' in bulleted points, 
          each with a short 1-2 sentence summary and a URL. Use find_news tool to find the news articles and then use the observation as action input to the final_result_tool.
          Ensure that all information is directly related to '{query}'."""


    if st.button("Search"):
        st.info("Searching for news articles...")
        news_articles = agent.run(prompt)
        
        # Display the search results
        with st.container():
            st.subheader("Searched Results:")
            st.markdown(news_articles)

if __name__ == "__main__":
    main()
