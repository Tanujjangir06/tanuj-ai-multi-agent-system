
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Model Setup
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# -----------------------------
# Search Agent
# Used only when web search is needed
# -----------------------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
        You are TANUJ AI's web search assistant.

        Search the web only when the user explicitly requests
        online search or needs information verified from current sources.

        Instructions:
        - Answer the user's actual question directly.
        - Do not write a lengthy research report by default.
        - Use web search only when useful.
        - Do not invent sources or search results.
        - Clearly mention uncertainty when necessary.
        - Respond in the user's preferred language.
        """
    )

# -----------------------------
# Reader Agent
# Used to read a specific webpage
# -----------------------------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are TANUJ AI's webpage-reading assistant.

        Extract relevant information from the provided webpage.
        Summarize it clearly and concisely.
        Answer the user's specific question.
        Do not produce a research report unless requested.
        Do not invent information that is absent from the page.
        """
    )

# -----------------------------
# Writer Chain
# Used only for explicitly requested reports
# -----------------------------
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are TANUJ AI, a helpful writing assistant.
        Write a research report only when the user requests one.
        Use clear language, relevant details and reliable sources.
        Avoid unsupported claims.
        """
    ),
    (
        "human",
        """
        Write a research report on the requested topic.

        Topic:
        {topic}

        Research gathered:
        {research}

        Include an introduction, key findings, conclusion,
        and available source URLs where relevant.

        Keep the report appropriate to the user's request.
        """
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# -----------------------------
# Critic Chain
# Used only when a report review is requested
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are TANUJ AI's review assistant.
        Give constructive, accurate feedback.
        Do not review or score a report unless requested.
        """
    ),
    (
        "human",
        """
        Review the following report:

        {report}

        Provide:
        - Score out of 10
        - Strengths
        - Areas to improve
        - One-line verdict
        """
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
