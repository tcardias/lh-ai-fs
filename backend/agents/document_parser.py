import json
import logging

from llm import async_call_llm
from models import ParsedDocument

logger = logging.getLogger(__name__)

_SYSTEM = """You are a legal document parser specializing in motions for summary judgment.

Your task is to extract three categories of content from the provided MSJ text:

1. **citations** — every legal citation (case law, statutes). For each, capture:
   - citation_id: sequential string like "cit_1", "cit_2", etc.
   - case_name: the full case name
   - reporter: the reporter abbreviation (e.g., "Cal.4th", "F.2d", "F.Supp.2d")
   - year: the year in parentheses (integer)
   - volume: the volume number before the reporter (integer or null)
   - start_page: the page number after the reporter (integer or null)
   - pin_page: the pinpoint page after the start page (integer or null)
   - court: the court in parentheses, if given (string or null)
   - proposition: the proposition the MSJ claims this citation supports (copy the surrounding sentence)
   - msj_section: the Roman numeral section heading where it appears

2. **quotes** — every passage enclosed in double quotation marks that is attributed to a case or document. For each:
   - quote_id: sequential string like "q_1", "q_2", etc.
   - attributed_to: the source named in the surrounding text
   - attributed_page: the page/pin number if given (string or null)
   - quoted_text: the exact text inside the quotation marks
   - msj_context: the full sentence containing the quote
   - msj_section: the Roman numeral section heading

3. **factual_assertions** — numbered facts in "Statement of Undisputed Material Facts" and any factual claims in the Argument. For each:
   - assertion_id: sequential string like "fa_1", "fa_2", etc.
   - msj_section: the section heading
   - text: the full text of the assertion
   - assertion_type: one of "date", "equipment", "employment", "legal_status", "procedural", "other"

Return ONLY a JSON object with keys "citations", "quotes", and "factual_assertions". No prose."""


async def agent_document_parser(msj_text: str) -> ParsedDocument:
    """Extract all citations, quotes, and factual assertions from an MSJ.

    Performs a single JSON-mode LLM call to parse the full document in one pass.
    Does no verification — that responsibility belongs to the downstream agents.

    Args:
        msj_text: Raw text of the Motion for Summary Judgment.

    Returns:
        A ``ParsedDocument`` containing all extracted elements.

    Failure modes:
        - Embedded footnote citations may be missed if treated as body text.
        - Paraphrased quotes (no quotation marks) will not appear in ``quotes``.
        - Multi-sentence assertions may be collapsed into one.
    """
    logger.info("DocumentParser starting — MSJ length: %d chars", len(msj_text))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"Parse this Motion for Summary Judgment:\n\n{msj_text}",
        },
    ]
    raw = await async_call_llm(messages, response_format={"type": "json_object"})
    result = ParsedDocument.model_validate(json.loads(raw))
    logger.info(
        "DocumentParser complete — %d citations, %d quotes, %d assertions",
        len(result.citations),
        len(result.quotes),
        len(result.factual_assertions),
    )
    return result
