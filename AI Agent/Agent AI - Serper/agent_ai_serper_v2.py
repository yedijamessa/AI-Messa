import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# Streamlit UI
st.title("AI Agent Blog Generator")
st.write("Use a custom prompt to generate research-based blog content.")

user_prompt = st.text_area("Enter your prompt for the agents (e.g., write about AI trends in healthcare):")
run_button = st.button("Run AI Agents")

if run_button and user_prompt:
    # Set up API keys (for production, use Streamlit Secrets)
    os.environ["SERPER_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    
    # Instantiate tools
    docs_tool = DirectoryReadTool(directory='./blog-posts')
    file_tool = FileReadTool()
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    openai_llm = LLM(model="gpt-4")

    # Create agents
    researcher = Agent(
        role='Market Research Analyst',
        goal='Provide up-to-date research and insights',
        backstory='An expert analyst with a keen eye for relevant trends.',
        tools=[search_tool, web_rag_tool],
        verbose=True
    )

    writer = Agent(
        role='Content Writer',
        goal='Craft engaging articles and blogs from provided research.',
        backstory='A skilled writer who makes complex information simple and accessible.',
        tools=[docs_tool, file_tool],
        verbose=True
    )

    editor = Agent(
        role='Editor',
        goal='Polish and format blog posts to ensure clarity, grammar, and markdown style.',
        backstory='An editorial expert who enhances readability and professionalism.',
        tools=[],
        llm=openai_llm,
        verbose=True
    )

    # Define tasks
    research = Task(
        description=f"Research task based on the following prompt: {user_prompt}",
        expected_output="A concise summary (3-5 paragraphs) with citations or references, if applicable.",
        agent=researcher
    )

    write = Task(
        description="Write a blog post based on the research summary. Include a clear structure with intro, body, and conclusion.",
        expected_output="A markdown blog post with 4 paragraphs, human-readable and jargon-free.",
        agent=writer,
        output_file='blog-posts/new_post.md'
    )

    edit = Task(
        description="Edit the blog post for clarity, flow, and formatting. Ensure the final version is suitable for publishing.",
        expected_output="A polished markdown blog file ready for website upload.",
        agent=editor
    )

    # Assemble a crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research, write, edit],
        verbose=True,
        planning=True
    )

    # Execute tasks
    result = crew.kickoff()

    # Display result
    st.success("Content generation complete!")
    st.markdown("### Agent Result")
    st.markdown(result)

    # Show the final blog post
    try:
        with open('blog-posts/new_post.md', 'r') as f:
            content = f.read()
            st.markdown("### Final Blog Post")
            st.markdown(content)
    except FileNotFoundError:
        st.warning("Blog post file not found.")
