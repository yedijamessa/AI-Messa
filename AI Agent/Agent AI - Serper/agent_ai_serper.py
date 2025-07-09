import os
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# Streamlit UI
st.title("AI Industry Blog Generator")

if st.button("Run AI Blog Generator"):
    # Set up API keys (in production, use Streamlit Secrets or secure environment vars)
    os.environ["SERPER_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""

    # Instantiate tools
    docs_tool = DirectoryReadTool(directory='./blog-posts')
    file_tool = FileReadTool()
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()

    # Create agents
    researcher = Agent(
        role='Market Research Analyst',
        goal='Provide up-to-date market analysis of the AI industry',
        backstory='An expert analyst with a keen eye for market trends.',
        tools=[search_tool, web_rag_tool],
        verbose=True
    )

    writer = Agent(
        role='Content Writer',
        goal='Craft engaging blog posts about the AI industry',
        backstory='A skilled writer with a passion for technology.',
        tools=[docs_tool, file_tool],
        verbose=True
    )

    # Define tasks
    research = Task(
        description='Research the latest trends in the AI industry and provide a summary.',
        expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
        agent=researcher
    )

    write = Task(
        description='Write an engaging blog post about the AI industry, based on the research analysts summary. Draw inspiration from the latest blog posts in the directory.',
        expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
        agent=writer,
        output_file='blog-posts/new_post.md'
    )

    # Assemble a crew with planning enabled
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research, write],
        verbose=True,
        planning=True,
    )

    # Execute tasks
    result = crew.kickoff()

    # Display result
    st.success("Blog generation complete!")
    st.markdown("### Result")
    st.markdown(result)

    # Optionally, display saved blog content
    try:
        with open('blog-posts/new_post.md', 'r') as f:
            content = f.read()
            st.markdown("### Generated Blog Post")
            st.markdown(content)
    except FileNotFoundError:
        st.warning("Blog post file not found.")
