"""
Core LLM Interface classes for different model providers.
Handles the low-level interaction with OpenAI, Google Gemini, and Anthropic APIs.
"""

import os
import re


class LLMClient:
    """Unified interface for different LLM providers."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = None
        self.last_request_parameters = None
        self.last_usage = self._usage(count_source="not_called")
        self._setup_client()

    def _provider(self):
        if self.model_name.startswith(("gpt-", "o-")):
            return "openai"
        if self.model_name.startswith("gemini-"):
            return "google"
        if self.model_name.startswith("claude-"):
            return "anthropic"
        return "unknown"

    @staticmethod
    def _field(value, name, default=None):
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _usage(
        self,
        *,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        cached_input_tokens=None,
        cache_creation_input_tokens=None,
        reasoning_tokens=None,
        count_source="provider_reported",
    ):
        return {
            "provider": self._provider(),
            "model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "count_source": count_source,
            "request_parameters": self.last_request_parameters,
        }

    def _supports_sampling_parameters(self):
        """Return whether this model accepts explicit temperature and top_p."""
        if self.model_name.startswith("gpt-5.6-"):
            return False
        if self.model_name.startswith(("gemini-3.5-", "gemini-3.6-")):
            return False
        if self.model_name.startswith((
            "claude-fable-",
            "claude-mythos-",
            "claude-opus-5",
            "claude-sonnet-5",
        )):
            return False
        opus_match = re.match(r"claude-opus-4-(\d+)", self.model_name)
        return not opus_match or int(opus_match.group(1)) < 7

    def _sampling_kwargs(self, temperature, top_p):
        if not self._supports_sampling_parameters():
            return {}
        return {"temperature": temperature, "top_p": top_p}

    def _request_parameters(self, max_tokens, sampling_kwargs):
        self.last_request_parameters = {
            "max_tokens": max_tokens,
            "temperature": sampling_kwargs.get("temperature"),
            "top_p": sampling_kwargs.get("top_p"),
        }
    
    def _setup_client(self):
        """Initialize the appropriate client based on model name."""
        if self.model_name.startswith("gpt-") or self.model_name.startswith("o-"):
            self._setup_openai_client()
        elif self.model_name.startswith("gemini-"):
            self._setup_gemini_client()
        elif self.model_name.startswith("claude-"):
            self._setup_anthropic_client()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _setup_openai_client(self):
        """Setup OpenAI client."""
        from openai import OpenAI
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=openai_api_key)
    
    def _setup_gemini_client(self):
        """Setup Gemini client."""
        from google import genai
        from google.genai import types
        
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_api_key is None:
            raise ValueError("API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY.")
        self.client = genai.Client(api_key=gemini_api_key)
        self.genai_types = types

    def _setup_anthropic_client(self):
        """Setup Anthropic client."""
        from anthropic import Anthropic

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key is None:
            raise ValueError("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        self.client = Anthropic(api_key=anthropic_api_key)
    
    def generate_content(self, prompt, system_prompt, max_tokens=1024, temperature=0.7, top_p=1.0):
        """Generate content using the appropriate model."""
        if self.model_name.startswith("gpt-") or self.model_name.startswith("o-"):
            return self._generate_openai_content(prompt, system_prompt, max_tokens, temperature, top_p)
        elif self.model_name.startswith("gemini-"):
            return self._generate_gemini_content(prompt, system_prompt, max_tokens, temperature, top_p)
        elif self.model_name.startswith("claude-"):
            return self._generate_anthropic_content(prompt, system_prompt, max_tokens, temperature, top_p)
    
    def _generate_openai_content(self, prompt, system_prompt, max_tokens, temperature, top_p):
        """Generate content using OpenAI models."""
        sampling_kwargs = self._sampling_kwargs(temperature, top_p)
        self._request_parameters(max_tokens, sampling_kwargs)
        token_limit = (
            {"max_completion_tokens": max_tokens}
            if self.model_name.startswith(("gpt-5", "o-"))
            else {"max_tokens": max_tokens}
        )
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            **token_limit,
            **sampling_kwargs,
            stream_options={"include_usage": True},
        )
        
        content_parts = []
        usage = None
        finish_reason = None
        
        for chunk in stream:
            chunk_usage = self._field(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage
            choices = self._field(chunk, "choices", []) or []
            if choices:
                chunk_finish_reason = self._field(choices[0], "finish_reason")
                if chunk_finish_reason is not None:
                    finish_reason = str(chunk_finish_reason)
                delta = self._field(choices[0], "delta")
                content = self._field(delta, "content", "") or ""
                content_parts.append(content)

        if usage is None:
            self.last_usage = self._usage(count_source="unavailable")
            return ''.join(content_parts), 0
        prompt_details = self._field(usage, "prompt_tokens_details")
        completion_details = self._field(usage, "completion_tokens_details")
        total = self._field(usage, "total_tokens")
        self.last_usage = self._usage(
            input_tokens=self._field(usage, "prompt_tokens"),
            output_tokens=self._field(usage, "completion_tokens"),
            total_tokens=total,
            cached_input_tokens=self._field(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=self._field(completion_details, "reasoning_tokens", 0) or 0,
        )
        self.last_usage["finish_reason"] = finish_reason
        self.last_usage["truncated"] = finish_reason == "length"
        return ''.join(content_parts), int(total or 0)
    
    def _generate_gemini_content(self, prompt, system_prompt, max_tokens, temperature, top_p):
        """Generate content using Gemini models."""
        sampling_kwargs = self._sampling_kwargs(temperature, top_p)
        self._request_parameters(max_tokens, sampling_kwargs)
        config = self.genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            **sampling_kwargs,
        )
        
        stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        
        content_parts = []
        usage = None
        finish_reason = None
        
        for chunk in stream:
            content = chunk.text
            if content:
                content_parts.append(content)
            chunk_usage = self._field(chunk, "usage_metadata")
            if chunk_usage is not None:
                usage = chunk_usage
            candidates = self._field(chunk, "candidates", []) or []
            if candidates:
                candidate_reason = self._field(candidates[0], "finish_reason")
                if candidate_reason is not None:
                    finish_reason = str(candidate_reason)
        
        full_content = ''.join(content_parts)
        if usage is None:
            self.last_usage = self._usage(count_source="unavailable")
            return full_content, 0
        total = self._field(usage, "total_token_count")
        self.last_usage = self._usage(
            input_tokens=self._field(usage, "prompt_token_count"),
            output_tokens=self._field(usage, "candidates_token_count"),
            total_tokens=total,
            cached_input_tokens=self._field(usage, "cached_content_token_count", 0) or 0,
            reasoning_tokens=self._field(usage, "thoughts_token_count", 0) or 0,
        )
        self.last_usage["finish_reason"] = finish_reason
        self.last_usage["truncated"] = bool(
            finish_reason and any(word in finish_reason.upper() for word in ("MAX_TOKENS", "LENGTH"))
        )
        return full_content, int(total or 0)

    def _generate_anthropic_content(self, prompt, system_prompt, max_tokens, temperature, top_p):
        """Generate content using Anthropic Claude models."""
        sampling_kwargs = self._sampling_kwargs(temperature, top_p)
        self._request_parameters(max_tokens, sampling_kwargs)
        with self.client.messages.stream(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **sampling_kwargs,
        ) as stream:
            content_parts = list(stream.text_stream)
            response = stream.get_final_message()
        usage = getattr(response, "usage", None)
        if usage is not None:
            uncached_input = self._field(usage, "input_tokens", 0) or 0
            cache_creation = self._field(usage, "cache_creation_input_tokens", 0) or 0
            cached_input = self._field(usage, "cache_read_input_tokens", 0) or 0
            output_tokens = self._field(usage, "output_tokens", 0) or 0
            input_tokens = uncached_input + cache_creation + cached_input
            total_tokens = input_tokens + output_tokens
            self.last_usage = self._usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input,
                cache_creation_input_tokens=cache_creation,
                reasoning_tokens=self._field(usage, "reasoning_tokens", 0) or 0,
            )
            finish_reason = self._field(response, "stop_reason")
            self.last_usage["finish_reason"] = finish_reason
            self.last_usage["truncated"] = finish_reason == "max_tokens"
        else:
            total_tokens = 0
            self.last_usage = self._usage(count_source="unavailable")
        return "".join(content_parts), total_tokens
