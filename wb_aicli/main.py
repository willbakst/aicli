"""The CLI Implementation."""
# --------------------------------------------------------------------------------
# Source: https://github.com/samuelcolvin/aicli/blob/main/samuelcolvin_aicli.py
# By: samuelcolvin
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
#
# The below code is a version of the source modified to fit this project's purpose.
# --------------------------------------------------------------------------------
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional, cast

from mirascope import OpenAIChat, Prompt, messages
from openai import OpenAIError
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text
from typer import Argument, Option, Typer

from . import __version__

app = Typer()


class SimpleCodeBlock(CodeBlock):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style="dim")
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color="default",
            word_wrap=True,
        )
        yield Text(f"/{self.lexer_name}", style="dim")


Markdown.elements["fence"] = SimpleCodeBlock


@messages
class ChatPrompt(Prompt):
    """
    SYSTEM:
    Help the user by responding to their request, the output should be concise and
    always written in markdown. The current date and time is {now}. The user is
    running {platform}.

    {formatted_context}

    USER:
    {user_prompt}
    """

    user_prompt: str
    context: list[tuple[str, str]] = []

    @property
    def formatted_context(self) -> str:
        """Returns the formatted context."""
        return "\n".join(
            [f"{role.upper()}: {content}" for role, content in self.context]
        )

    @property
    def now(self) -> str:
        """Returns the current date and time."""
        now_utc = datetime.now(timezone.utc)
        tzinfo = now_utc.astimezone().tzinfo
        if tzinfo is None:
            return f"{datetime.now()}"
        else:
            return f"{datetime.now()} {tzinfo.tzname(now_utc)}"

    @property
    def platform(self) -> str:
        """Returns the user's platform."""
        return f"{sys.platform}"


def _create(
    chat: OpenAIChat, user_prompt: str, context: list[tuple[str, str]], console: Console
) -> Optional[str]:
    """Creates a response from the user's prompt."""
    error = None
    with Status("[dim]Working on it…[/dim]", console=console):
        try:
            prompt = ChatPrompt(user_prompt=user_prompt, context=context)
            completion = chat.create(prompt)
        except OpenAIError as e:
            error = e

    if error:
        console.print(f"OpenAI error: {error}", style="red")
        context.append(("assistant", f"OpenAI error: {error}"))
        return None
    else:
        console.print(Markdown(str(completion)))
        context.append(("assistant", str(completion)))
        return str(completion)


def _stream(
    chat: OpenAIChat, user_prompt: str, context: list[tuple[str, str]], console: Console
):
    """Streams the response from the user's prompt."""
    interrupted, error = False, None
    content = ""
    with Live("", refresh_per_second=15, console=console) as live:
        try:
            prompt = ChatPrompt(user_prompt=user_prompt, context=context)
            for chunk in chat.stream(prompt):
                if chunk.choice.finish_reason is not None:
                    break
                content += str(chunk)
                live.update(Markdown(content))
        except OpenAIError as e:
            error = e
        except KeyboardInterrupt:
            interrupted = True

    if error:
        console.print(f"OpenAI error: {error}", style="red")

    if interrupted:
        console.print("[dim]Interrupted[/dim]")

    context.append(("assistant", content))


def _interactive_mode(chat: OpenAIChat, stream: bool, console: Console):
    """Runs an interactive CLI mode."""
    history = Path().home() / ".openai-prompt-history.txt"
    session = PromptSession(history=FileHistory(str(history)))  # type: ignore
    multiline = False

    context = [ChatPrompt(user_prompt="").messages[0]]
    while True:
        try:
            text = session.prompt(
                "wb: aicli ➤ ",
                auto_suggest=AutoSuggestFromHistory(),
                multiline=multiline,
            )
        except (KeyboardInterrupt, EOFError):
            return

        if not text.strip():
            continue

        ident_prompt = text.lower().strip(" ").replace(" ", "-")
        if ident_prompt == "show-markdown":
            last_content = context[-1][1]
            console.print("[dim]Last markdown output of last question:[/dim]\n")
            console.print(
                Syntax(last_content, lexer="markdown", background_color="default")
            )
            continue
        elif ident_prompt == "multiline":
            multiline = not multiline
            if multiline:
                console.print(
                    "Enabling multiline mode. "
                    "[dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept "
                    "input.[/dim]"
                )
            else:
                console.print("Disabling multiline mode.")
            continue

        context.append(("user", text))
        console.print("\nResponse:", style="green")
        try:
            if stream:
                _stream(chat, text, context, console)
            else:
                _create(chat, text, context, console)  # type: ignore
        except KeyboardInterrupt:
            return


@app.command(help="Chat with an AI model")
def chat(
    user_prompt: Annotated[str, Argument(help="Interactive mode if empty")] = "",
    model: Annotated[str, Option(help="The model to use")] = "gpt-3.5-turbo-1106",
    base_url: Annotated[Optional[str], Option(help="The base URL for the API")] = None,
    stream: Annotated[bool, Option(help="Streams responses")] = False,
):
    console, api_key = startup()
    if not api_key:
        return

    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        console.print(
            "You must set the OPENAI_API_KEY environment variable", style="red"
        )
        return

    chat = OpenAIChat(model=model, api_key=openai_api_key, base_url=base_url)
    if user_prompt:
        console.print("\nResponse:", style="green")
        try:
            if stream:
                _stream(chat, user_prompt, [], console)
            else:
                _create(chat, user_prompt, [], console)
        except KeyboardInterrupt:
            pass
        return
    else:
        _interactive_mode(chat, stream, console)


class DesiredFieldInfos(BaseModel):
    """The arguments for creating a pydantic `BaseModel` with `create_model` method."""

    field_names: list[str] = Field(..., description="The desired field names.")
    types: list[str] = Field(
        ...,
        description="The desired types for the fields e.g. <class str>.",
    )
    descriptions: list[str] = Field(
        ..., description="The descriptions for the desired fields."
    )


@messages
class GenerateModelPrompt(Prompt):
    """
    SYSTEM:
    Your task is to identify the desired fields for a pydantic `BaseModel` based on a
    user query. You will then generate the tool calls with the identified arguments.
    Your output should be a tool call with the field definitions argument.

    USER:
    {query}
    """

    query: str


@messages
class ExtractInfoPrompt(Prompt):
    """
    SYSTEM:
    Your task is to extract information from a provided file. You will receive the file
    as well as tool schemas defining what information to extract.

    USER:
    Here is the file:

    {file}
    """

    file: str


@app.command(help="Extract information from a file")
def extract(
    filename: Annotated[str, Argument(help="The file for extraction.")],
    query: Annotated[str, Argument(help="The query for extraction.")],
    model: Annotated[str, Option(help="The model to use")] = "gpt-3.5-turbo-1106",
    base_url: Annotated[Optional[str], Option(help="The base URL for the API")] = None,
):
    """Extracts desired information from a target file."""
    if not os.path.exists(filename):
        raise FileNotFoundError("{filename} not found for extraction")

    with open(filename, "r") as file:
        file_contents = file.read()

    # First, we need to dynamically generate a schema
    console, api_key = startup()
    if not api_key:
        return

    chat = OpenAIChat(model=model, api_key=api_key, base_url=base_url)
    try:
        console.print("\nResponse:", style="green")
        with Status("[dim]Generating schema for extraction…[/dim]", console=console):
            desired_info = chat.extract(
                DesiredFieldInfos, GenerateModelPrompt(query=query)
            )
            field_definitions: dict[str, Any] = {
                field_name: (
                    t,
                    FieldInfo(annotation=cast(Any, t), description=description),
                )
                for field_name, t, description in zip(
                    desired_info.field_names,
                    desired_info.types,
                    desired_info.descriptions,
                )
            }
            base_model_type = create_model(
                "ExtractedFields",
                __doc__="The fields to extract from a given file.",
                **field_definitions,
            )

        with Status("[dim]Extracting information from file…[/dim]", console=console):
            extracted_info = chat.extract(
                base_model_type, ExtractInfoPrompt(file=file_contents)
            )

        console.print(f"Extracted Info: {extracted_info}")
    except ValidationError as e:
        console.print(f"ValidationError error: {e}", style="red")
        return
    except OpenAIError as e:
        console.print(f"OpenAI error: {e}", style="red")
        return
    except KeyboardInterrupt:
        return


@app.command(help="Show the version")
def version():
    startup(with_key=False)


def startup(with_key: bool = True) -> tuple[Console, Optional[str]]:
    """Prints startup message and returns console."""
    console = Console()
    console.print(
        f"William Bakst's copy-cat of aicli - OpenAI powered AI CLI v{__version__}",
        style="green bold",
        highlight=False,
    )
    openai_api_key = None

    if with_key:
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            console.print(
                "You must set the OPENAI_API_KEY environment variable", style="red"
            )

    return console, openai_api_key


if __name__ == "__main__":
    app()
