from openai import OpenAI

client = OpenAI(
    api_key=""
)

def query_llm(model: str, prompt: str, dataset_instructions: str) -> str:
    """Send a prompt to the specified OpenAI LLM model and return the response."""

    response = client.responses.create(
        model=model,
        instructions="""You're interacting with another LLM to complete this task - to answer
            the given question. {0} Please keep your responses very brief. Your answers
            will be given to the other LLM to help inform their prediction,
            and then their answers will be given to you to help inform your prediction. The
            other LLM has been given the same instrutions. The prompts you will receive are
            the given question and then the transcript of the conversation between you and
            the other LLM, which is the sequence of predictions they make.""".format(dataset_instructions),
        input=prompt
    )

    return response.output_text

def interactive_llm_loop(
    initial_prompt: str,
    dataset_instructions: str,
    model_a: str = "gpt-4.1-nano",
    model_b: str = "gpt-4o-mini",
    max_turns: int = 5,
):
    """Alternate queries between two LLMs, feeding responses into the other."""
    transcript = initial_prompt
    current_prompt = transcript # Current prompt is full transcript
    for turn in range(max_turns):
        if turn % 2 == 0:
            # Model A's turn
            print(f"\nRound {turn} — Model A ({model_a})")
            response = query_llm(model_a, current_prompt, dataset_instructions)
        else:
            # Model B's turn
            print(f"\nTurn {turn} — Model B ({model_b})")
            response = query_llm(model_b, current_prompt, dataset_instructions)
        transcript += f"\nTurn {turn} Output\n {response}\n"

        print("Transcript:\n", transcript)

        print("Response:\n", response)

if __name__ == "__main__":
    starting_prompt="""Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18))
        over Q. The answer choices are [0, 4, 2, 6]."""
    dataset_instructions = """The question is a multiple choice question with four answer choices.
        Please output the probability you think each answer is the correct one, in the same order
        as the given choices."""

    interactive_llm_loop(starting_prompt, dataset_instructions)

