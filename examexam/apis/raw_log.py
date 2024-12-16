def format_conversation_to_markdown(
    conversation: list[dict[str, str]], user_label: str = "User", assistant_label: str = "Assistant"
) -> str:
    """
    Formats a conversation (a list of message dictionaries) into a Markdown string.

    Args:
        conversation (List[Dict[str, str]]): The conversation to format. Each message in the
                                              conversation is a dictionary with 'role' and 'content' keys.
        user_label (str, optional): The label to use for the user's messages. Defaults to "User".
        assistant_label (str, optional): The label to use for the assistant's messages. Defaults to "Assistant".

    Returns:
        str: The formatted conversation as a Markdown string.
    """
    markdown_lines = []

    for message in conversation:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")

        if role.lower() == "user":
            label = user_label
        elif role.lower() == "assistant":
            label = assistant_label
        elif role.lower() == "examexam":
            label = "LLM Build Error Message"
        elif role.lower() == "system":
            label = "System"
        else:
            label = role  # For 'system' and any other roles that might be present

        # Make invisible characters visible
        if content is None:
            content = f"**** {label} returned None, maybe API failed ****"
        elif content.strip() == "":
            content = f"**** {label} returned whitespace ****"
        elif not content:
            content = f"**** {label} returned falsy value {content} ****"

        markdown_line = f"{label}: {content}"
        markdown_lines.append(markdown_line)

    return "\n".join(markdown_lines)
