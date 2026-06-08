from src.tokenizer import Tokenizer

def test_tokenizer_decoding() -> None:
    print(">>> Initializing Tokenizer...")
    try:
        tokenizer = Tokenizer()
    except Exception as e:
        print(f">>> ERROR: Failed to initialize Tokenizer. Is the SDK configured? ({e})")
        return

    target_string = "What is the sum of 2 and 3?"

    # 1. Use the SDK's true encoder to get Qwen3's actual token IDs
    # The SDK returns a 2D PyTorch tensor, so we extract the first row and convert to a list
    real_ids_tensor = tokenizer.sdk.encode(target_string)
    real_ids = real_ids_tensor[0].tolist()

    print(f">>> Real Qwen3 IDs: {real_ids}")

    # 2. Feed the real IDs into YOUR custom decoder
    decoded_text = tokenizer.decode(real_ids)

    print(f">>> Expected : '{target_string}'")
    print(f">>> Result   : '{decoded_text}'")

    if decoded_text == target_string:
        print("✅ SUCCESS: Byte-level decoding is perfectly aligned.")
    else:
        print("❌ FAILED: The output does not match the expected string.")

def test_tokenizer_encoding() -> None:
    print(">>> Initializing Tokenizer...")
    tokenizer = Tokenizer()

    target_string = "I'm playing football"
    print(f"\n>>> Target String: '{target_string}'")

    # 1. Official SDK Encoding
    sdk_ids_tensor = tokenizer.sdk.encode(target_string)
    sdk_ids = sdk_ids_tensor[0].tolist()
    print(f">>> Official SDK IDs: {sdk_ids}")

    # 2. Your Custom Greedy Encoding
    custom_ids = tokenizer.encode(target_string)
    print(f">>> Custom Greedy IDs: {custom_ids}")

    if sdk_ids == custom_ids:
        print("✅ SUCCESS: Custom encoder matches the official SDK perfectly.")
    else:
        print("⚠️ NOTE: IDs differ. The greedy algorithm made different chunking choices than Qwen's merge rules.")

    # Verify your IDs can still decode properly
    print(f"\n>>> Verifying custom IDs decode correctly: '{tokenizer.decode(custom_ids)}'")

if __name__ == "__main__":
    test_tokenizer_encoding()
