"""Metadata for examexam."""

__all__ = [
    "__credits__",
    "__dependencies__",
    "__description__",
    "__readme__",
    "__requires_python__",
    "__status__",
    "__title__",
    "__version__",
]

__title__ = "examexam"
__version__ = "0.1.9"
__description__ = (
    "Let LLM generate tests, validate test, and then you take them. Useful for certs."
)
__credits__ = [{"name": "Matthew Martin", "email": "matthewdeanmartin@gmail.com"}]
__readme__ = "README.md"
__requires_python__ = ">=3.11"
__dependencies__ = [
    "Jinja2",
    "rtoml>=0.1.0; python_version < '3.15'",
    "rtoml>=0.14.0; python_version >= '3.15'",
    "tomli-w>=1.2.0; python_version >= '3.15'",
    "scipy",
    "orjson",
    "argcomplete",
    "packaging",
    "do_i_need_to_upgrade>=0.1.0",
    "markdown",
    "anthropic>=0.19.2",
    "google-generativeai>=0.8.4",
    "openai>=1.0.0",
    "retry>=0.9.2",
    "colorlog>=6.8.2",
    "rich>=13.0.0",
    "python-dotenv>=1.0.1",
]
__status__ = "4 - Beta"
