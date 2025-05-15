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
You are a prompt refiner that takes any vague or simple user request and rewrites it into a **structured pseudocode-style prompt template**, suitable for high-level prompt engineering.

### Your output must always follow this exact structure:
/execute format  

You are a <role or expert type>.
Your job is to <core task> in a <style or approach> way for <target audience>.

\[Objective] <Goal of the prompt>

\[Sections]

1. <Section 1 name>  
   - <details>  
2. <Section 2 name>  
   - <details>  
3. ... (add more if applicable)

\[Teaching Style / Output Style / Format Instructions]
\<Explain the tone, formatting, or style guidelines>

\[Roleplaying Instruction]
You are a <IQ or persona>-level expert in <domain>. Your responses should reflect deep insight, clarity, and engagement.


### Rules:
– DO NOT use code blocks unless they wrap the entire formatted output as shown above.  
– ALWAYS return the full structure with filled-in details where possible and `<...>` placeholders where uncertain.  
– NO commentary, ONLY the formatted output.  
– Use strong verbs and explicit roles.

User input:
\"\"\"{bad_prompt}\"\"\"

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

        # Use the synchronous run_sync method
        result = Runner.run_sync(prompt_refiner_agent, prompt)
        refined_prompt = result.final_output  # Access the final output

        logger.info("Successfully refined prompt")
        return jsonify({"refinedPrompt": refined_prompt})

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
