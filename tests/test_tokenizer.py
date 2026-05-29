from src.tokenizer import Tokenizer

def run_test() -> None:
    print(">>> Initializing Tokenizer...")
    try:
        tokenizer = Tokenizer()
    except Exception as e:
        print(f">>> ERROR: Failed to initialize Tokenizer. Is the SDK configured? ({e})")
        return

    # Illustrative Qwen token IDs from the subject:
    # Maps roughly to: ["What", "Ġis", "Ġthe", "Ġsum", "Ġof", "Ġ2", "Ġand", "Ġ3", "?"]
    test_ids = [892, 318, 262, 4771, 286, 16, 290, 17, 30]

    print(f">>> Decoding IDs: {test_ids}")

    decoded_text = tokenizer.decode(test_ids)

    print(f">>> Expected : 'What is the sum of 2 and 3?'")
    print(f">>> Result   : '{decoded_text}'")

    if decoded_text == "What is the sum of 2 and 3?":
        print("✅ SUCCESS: Byte-level decoding is perfectly aligned.")
    else:
        print("❌ FAILED: The output does not match the expected string. Check the bytearray mapping.")

if __name__ == "__main__":
    run_test()
