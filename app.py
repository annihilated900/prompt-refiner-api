from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio
from openai import OpenAI
from agents import Agent, Runner, function_tool, ModelSettings
from dotenv import load_dotenv
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client once
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define your agent code
@function_tool
async def refine_prompt_tool(bad_prompt: str) -> str:
    meta_prompt = """
    You are a prompt‑refiner that takes any user request and emits a "pseudocode prompt skeleton" in the exact format below.
– Input: A user's original prompt.
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
            <instructions for c2 derived from user's goal>
        [END]

    [c3, Args: …]
        [INSTRUCTIONS]
            <instructions for c3 inferred from user's request>
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
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": meta_prompt},
                {"role": "user", "content": bad_prompt}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"Error calling OpenAI API: {str(e)}")

# Initialize your agent
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

# Add the API endpoint - NOTE: This is a regular function, not async
@app.route('/refine-prompt', methods=['POST'])
def refine_prompt():
    # Get the prompt from the request
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "Request must include JSON data"}), 400
            
        prompt = data.get('prompt')
        if not prompt:
            logger.warning("No prompt in request data")
            return jsonify({"error": "Prompt is required"}), 400
        
        logger.info(f"Received prompt: {prompt[:50]}...")
        
        # Run the async function in a synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(runner.run_agent(prompt_refiner_agent, prompt))
            refined_prompt = result.messages[-1].content
            logger.info("Successfully refined prompt")
            return jsonify({"refinedPrompt": refined_prompt})
        finally:
            loop.close()
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error processing request: {str(e)}\n{error_details}")
        return jsonify({
            "error": "Failed to process prompt",
            "details": str(e),
            "trace": error_details
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
