import time
from src.engine import ConstrainedDecoder
from src.utils import Formatter, error, warning

def main() -> None:
    print(Formatter.apply('bold', 'yellow',
                          "\n>>> Initializing Constrained Decoder Engine..."))
    start_time = time.time()

    try:
        engine = ConstrainedDecoder(model_name="Qwen/Qwen3-0.6B")
        elaps = time.time() - start_time
        print(Formatter.apply('bold', 'cyan',
                              f">>> Engine loaded in {elaps:.2f} seconds.\n"))

    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    # MOCK PARSER DATA: Bypassing the parser module temporarily
    mock_functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., Paris, France"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }
        }
    ]

    user_prompt = "Can you tell me how cold it is in Paris today? Use Celsius."

    print(Formatter.apply('bold', 'blue', f">>> 👤 User Prompt: ") + user_prompt)
    print(Formatter.apply('bold', 'yellow',
                          ">>> Intercepting logits and generating constrained JSON...\n"))
    print(Formatter.apply(None, 'gray', "-" * 60))

    try:
        # Execute the constrained decoding loop
        generation_start = time.time()
        result = engine.generate_function_call(
            user_prompt=user_prompt,
            functions=mock_functions,
            max_new_tokens=120
        )

        # Display the guaranteed valid JSON in striking lime green
        print(Formatter.apply('bold', 'lime', result))
        print(Formatter.apply(None, 'gray', "-" * 60))

        gen_time = time.time() - generation_start
        print(Formatter.apply('bold', 'cyan',
                              f">>>  Generation completed in {gen_time:.2f} seconds.\n"))

    except Exception as e:
        error(f"Generation loop failed: {e}")

if __name__ == "__main__":
    main()
