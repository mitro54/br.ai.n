import asyncio

import httpx

ROUTER_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434"

async def test_prompts():
    prompts = ["hello", "identify yourself", "write a python script to scrape a website", "what is 2+2?", "generate an image of a cat"]
    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            payload = {
                "model": ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "Classify the user intent into exactly one category: FAST (greetings, simple questions), GLM (reasoning, logic, brief history), EXPERT (coding, complex math, multi-step logic), or IMAGE (visual generation). Reply with ONLY the category name."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "keep_alive": -1
            }
            resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
            intent = resp.json()["choices"][0]["message"]["content"].strip().upper()
            print(f"Prompt: {prompt} -> Intent: {intent}")

if __name__ == "__main__":
    asyncio.run(test_prompts())
