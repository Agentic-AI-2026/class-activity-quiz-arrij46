'''
Arrij Fawwad 
22i0755
Section A
Agentic Quiz 
'''

from graph import build_graph, AgentState

def run(goal: str):
    graph = build_graph()
    initial_state: AgentState = {
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": []
    }

    print(f"\nGoal: {goal}\n")
    final_state = graph.invoke(initial_state)

    print(f"Plan ({len(final_state['plan'])} steps):")
    for s in final_state["plan"]:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()

    print("Results:")
    for r in final_state["results"]:
        print(f"  Step {r['step']}: {r['description']}")
        print(f"    {str(r['result'])[:300]}\n")

    return final_state["results"]


if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your goal: ")
    run(goal)
