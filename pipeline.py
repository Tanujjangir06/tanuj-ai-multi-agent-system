from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


def run_research(topic):
    # 1. Search Agent
    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Search the web for reliable information about: {topic}"
            }
        ]
    })

    # Get search output
    research = search_result["messages"][-1].content

    # 2. Reader Agent
    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Read and analyze the following research:\n\n{research}"
            }
        ]
    })

    research = reader_result["messages"][-1].content

    # 3. Writer Chain
    report = writer_chain.invoke({
        "topic": topic,
        "research": research
    })

    # 4. Critic Chain
    critique = critic_chain.invoke({
        "report": report
    })

    return report, critique