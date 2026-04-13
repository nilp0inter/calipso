from calipso.agent import agent


def main():
    result = agent.run_sync("Hello!")
    print(result.output)
