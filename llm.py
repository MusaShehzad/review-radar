"""Thin wrappers around the two LLM providers, so classify.py doesn't care which one it talks to.

Each client has one method, generate(system, prompt, schema), which returns the model's JSON
answer as text. Provider errors are translated into four kinds that classify.py handles:

    RateLimited     slow down and retry, or stop for today if the daily quota is used up
    TemporaryError  the service had a hiccup: wait and retry
    BadResponse     the model answered, but the answer can't be used (blocked, cut off, empty)
    FatalError      retrying won't help (bad API key, unknown model): stop and explain
"""
import os
import re

import httpx


class RateLimited(Exception):
    def __init__(self, message: str, wait_seconds: float | None = None, daily: bool = False):
        super().__init__(message)
        self.wait_seconds = wait_seconds  # how long the API asked us to wait, if it said
        self.daily = daily                # True when the daily quota is used up


class TemporaryError(Exception):
    pass


class BadResponse(Exception):
    pass


class FatalError(Exception):
    pass


def read_api_key(env_var: str) -> str:
    """Read an API key from the environment (classify.py loads .env into it first)."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise FatalError(f"No API key found. Paste your key after {env_var}= in the .env file.")
    return key


class GeminiClient:
    """Google Gemini, through the google-genai SDK."""

    def __init__(self, settings: dict):
        # Imported here, so using one provider doesn't require the other provider's SDK.
        from google import genai
        from google.genai import errors, types

        self.errors, self.types = errors, types
        self.model = settings["model"]
        self.thinking_level = settings.get("thinking_level")
        self.client = genai.Client(api_key=read_api_key(settings["api_key_env"]))

    def generate(self, system: str, prompt: str, schema: dict) -> str:
        thinking = None
        if self.thinking_level:
            thinking = self.types.ThinkingConfig(thinking_level=self.thinking_level.upper())
        config = self.types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,  # the answer must match this JSON schema
            thinking_config=thinking,
            # We pass no tools, so turn off the SDK's tool-calling helper (it prints a warning otherwise).
            automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except self.errors.APIError as error:
            raise self._translate(error) from error
        except httpx.TransportError as error:  # network trouble or a timeout
            raise TemporaryError(f"network error: {error}") from error

        if not response.text:
            reason = "no text"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = f"blocked: {response.prompt_feedback.block_reason}"
            elif response.candidates:
                reason = f"finish reason: {response.candidates[0].finish_reason}"
            raise BadResponse(f"empty answer ({reason})")
        return response.text

    def _translate(self, error) -> Exception:
        details = str(error.details)
        if error.code == 429:
            # Gemini says how long to wait ("retryDelay": "37s") and which quota ran out.
            delay = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", details)
            return RateLimited(
                "Gemini rate limit",
                wait_seconds=float(delay.group(1)) if delay else None,
                daily="PerDay" in details,
            )
        if error.code in (401, 403) or "API_KEY_INVALID" in details:
            return FatalError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.")
        if error.code == 404:
            return FatalError(f"Gemini doesn't know the model '{self.model}'. Check config/classifier.yaml.")
        if error.code >= 500:
            return TemporaryError(f"Gemini server error {error.code}")
        return FatalError(f"Gemini error {error.code}: {error.message}")


class ClaudeClient:
    """Anthropic Claude, through the anthropic SDK."""

    def __init__(self, settings: dict):
        import anthropic

        self.anthropic = anthropic
        self.model = settings["model"]
        self.effort = settings.get("effort")
        self.refusal_fallback = settings.get("refusal_fallback", False)
        # max_retries=0: classify.py does the waiting and retrying, the same way for both providers.
        self.client = anthropic.Anthropic(api_key=read_api_key(settings["api_key_env"]), max_retries=0)

    def generate(self, system: str, prompt: str, schema: dict) -> str:
        anthropic = self.anthropic
        output_config = {"format": {"type": "json_schema", "schema": schema}}  # answer must match the schema
        if self.effort:
            output_config["effort"] = self.effort
        fallback = {}
        if self.refusal_fallback:
            # If Claude's safety checks decline a batch, the API re-runs it on Anthropic's
            # recommended fallback model instead of returning the refusal.
            fallback = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                **fallback,
            )
        # Most specific errors first: the first four are all kinds of APIStatusError.
        except anthropic.RateLimitError as error:
            retry_after = error.response.headers.get("retry-after", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else None
            raise RateLimited("Claude rate limit", wait_seconds=wait) from error
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
            raise FatalError("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.") from error
        except anthropic.NotFoundError as error:
            raise FatalError(f"Claude doesn't know the model '{self.model}'. Check config/classifier.yaml.") from error
        except anthropic.APIStatusError as error:
            if error.status_code >= 500:  # includes 529 "overloaded"
                raise TemporaryError(f"Claude server error {error.status_code}") from error
            raise FatalError(f"Claude error {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:  # network trouble or a timeout
            raise TemporaryError(f"network error: {error}") from error

        if response.stop_reason == "refusal":
            raise BadResponse("Claude declined this batch")
        if response.stop_reason == "max_tokens":
            raise BadResponse("answer was cut off")
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            raise BadResponse("empty answer")
        return text


def make_client(provider: str, settings: dict):
    """Create the client for the provider named in config/classifier.yaml."""
    clients = {"gemini": GeminiClient, "claude": ClaudeClient}
    if provider not in clients:
        raise FatalError(f"Unknown provider '{provider}' in config/classifier.yaml (use gemini or claude).")
    return clients[provider](settings)
