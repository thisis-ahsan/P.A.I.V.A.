import ollama
import datetime
import AppOpener


system_prompt = "You're Annie, a helpful AI assistant developed by and for Ahsan (pronunced as eh-sun) Ansari. Ahsan is an Indian Teen Entrepreneur, Full Stack Developer, Python Developer, Marketing and AI Expert. Your job is to help Ahsan with every possible thing. You should sound like a real human as much as possible"

chat = [
    {
        "role": "system",
        "content": system_prompt,
    }
]


def gettime():
    """Return the current local time."""
    print("Fetching latest date and time...")
    return datetime.datetime.now().strftime("%H:%M:%S")


def openapp(app_name):
    """Open an application."""
    try:
        AppOpener.open(app_name)
        return f"Application '{app_name}' opened successfully."
    except Exception as e:
        return f"Failed to open {app_name}: {e}"


def closeapp(app_name):
    """Close an application."""
    try:
        AppOpener.close(app_name)
        return f"Application '{app_name}' closed successfully."
    except Exception as e:
        return f"Failed to close {app_name}: {e}"


# Tool definitions sent to Ollama
tools = [openapp, closeapp, gettime]

# Python functions executed when a tool is requested
available_tools = {
    "openapp": openapp,
    "closeapp": closeapp,
    "gettime": gettime,
}


def think(user_input):
    chat.append({
        "role": "user",
        "content": user_input,
    })

    while True:
        response = ollama.chat(
            model="qwen3.5:2b",
            messages=chat,
            tools=tools,
        )

        # Preserve the assistant response and its tool calls.
        chat.append(response.message)

        # If there are no tool calls, this is the final answer.
        if not response.message.tool_calls:
            answer = response.message.content
            print("Annie:", answer)
            return answer

        # Execute all tool calls before asking the model again.
        for tc in response.message.tool_calls:
            tool_name = tc.function.name
            arguments = tc.function.arguments

            print(f"Executing tool: {tool_name}")
            print(f"Arguments: {arguments}")

            if tool_name not in available_tools:
                result = f"Error: Unknown tool '{tool_name}'."
            else:
                try:
                    result = available_tools[tool_name](**arguments)
                except Exception as e:
                    result = f"Tool execution failed: {e}"

            print(f"Tool result: {result}")

            chat.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": str(result),
            })