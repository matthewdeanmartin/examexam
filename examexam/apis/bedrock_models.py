from dataclasses import dataclass, field


@dataclass
class TitanAnswers:
    token_count: int
    output_text: str
    completion_reason: str
    """FINISHED or LENGTH"""


@dataclass
class TitanResponse:
    input_text_token_count: int
    results: list[TitanAnswers] = field(default_factory=list)
