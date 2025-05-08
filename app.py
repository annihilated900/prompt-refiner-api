from flask import Flask, request, jsonify
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Define your agent code (from your provided example)
@function_tool
async def refine_prompt_tool(bad_prompt: str) -> str:
    meta_prompt = """
    You are a prompt‑refiner that takes any user request and emits a “pseudocode prompt skeleton” in the exact format below.
– Input: A user’s original prompt.
– Output: The same skeleton structure, with placeholders (<…>) filled where you can infer specifics, or left blank otherwise.

[Commands - Prefix: "/"]
    text: Execute format <text>
    c2: Execute <c2>
    c3: Execute <c3>
    c4: Execute <c4>
    c5: Execute <c5>

[Function Rules]
    1. Act as if you are executing code.
    2. Do not say: [INSTRUCTIONS], [BEGIN], [END], [IF], [ENDIF], [ELSEIF]
    3. Do not write in codeblocks when creating the curriculum.
    4. Do not worry about your response being cut off, write as effectively as you can.

[Functions]
    [say, Args: text]
        [BEGIN]
            You must strictly say and only say word‑by‑word <text> while filling out the <…> with the appropriate information.
        [END]

    [c2, Args: …]
        [BEGIN]
            <instructions for c2 derived from user’s goal>
        [END]

    [c3, Args: …]
        [INSTRUCTIONS]
            <instructions for c3 inferred from user’s request>
        [BEGIN]
            <execution pseudocode for c3>
        [END]

    [c4, Args: …]
        [INSTRUCTIONS]
            <instructions for c4>
        [BEGIN]
            <execution pseudocode for c4>
        [END]

    [c5, Args: …]
        [INSTRUCTIONS]
            <instructions for c5>
        [BEGIN]
            <execution pseudocode for c5>
        [END]

[Init]
    [BEGIN]
        <any initialization steps—infer variables, greetings>
    [END]

execute <Init>
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": meta_prompt},
            {"role": "user",   "content": bad_prompt}
        ]
    )
    return resp.choices[0].message.content

# Initialize your agent
from agents import ModelSettings

prompt_refiner_agent = Agent(
    name="Prompt Refiner",
    instructions=(
        "You are a prompt‑refiner agent. Always call refine_prompt_tool."
        "When given a rough user request, you produce a fully‑formed pseudocode prompt skeleton."
    ),
    tools=[refine_prompt_tool],
    model_settings=ModelSettings(tool_choice="required"),
    tool_use_behavior="stop_on_first_tool",
)

# Create a runner for the agent
runner = Runner()

# Add the API endpoint
@app.route('/refine-prompt', methods=['POST'])
async def refine_prompt():
    # Get the prompt from the request
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Call the agent to refine the prompt
    try:
        result = await runner.run_agent(prompt_refiner_agent, prompt)
        refined_prompt = result.messages[-1].content
        return jsonify({"refinedPrompt": refined_prompt})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
