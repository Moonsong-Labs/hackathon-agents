import yaml
import re
from string import Template


def load_prompt_config(prompt: str) -> str:
    """Load a prompt from a file."""
    with open(prompt, "r") as f:
        content = f.read()

        return parse_prompt_config(content)


def parse_prompt_template(body: str, tag_name: str) -> str:
    pattern = r"<{tag_name}>(.*?)</{tag_name}>".format(tag_name=tag_name)
    match = re.search(pattern, body, re.DOTALL)
    if not match:
        raise ValueError(
            "Failed to find <{tag_name}> tags in the prompt body.".format(
                tag_name=tag_name
            )
        )

    prompt_template = match.group(1).strip()
    # Escape $ signs
    prompt_template = prompt_template.replace("$", "$$")
    # Replace double curly braces with the ${variable} convention
    prompt_template = re.sub(r"\{\{\s*(.*?)\s*\}\}", r"${\1}", prompt_template)
    return prompt_template


def parse_prompt_config(content: str) -> dict:
    """Parse the prompt content into a dictionary."""
    try:
        header, body = content.split("---", 2)[1:]
        res = yaml.safe_load(header)

        # Extract the prompt templates within <tag_name> tags
        res['prompt_templates'] = {}
        tag_pattern = r"<(\w+)>.*?</\1>"
        tags = re.findall(tag_pattern, body, re.DOTALL)
        for tag in tags:
            res['prompt_templates'][tag] = parse_prompt_template(body, tag)

        return res

    except Exception as e:
        raise ValueError(f"Failed to parse prompt: {e}")
