# Refactor generations
- Bot must indicate the explanation along with the possible answer. It too often generats the answer list and expression list in different order

## Test validation
- Fix answer parsing - DONE
  - Picking 2 answers when only 1 expected- DONE
  - Failure to parse - DONE
    - Answers with commas!- DONE

## Test generation
- Give it some screen scrapings as ground truth
- Add (Select 1) to all questions without "(Select"
- Fix (Select n) when len(answers) != n

## Test taking
- Some way to mark question as dodgy. - 1/2 DONE
- Some visual when there are alternate answers - 1/2 DONE
- Grade as you go... - DONE
- Resumable test session - 1/2 DONE
- Option to report a question as stupid - 1/2 DONE

## Bad questions
- Trivia: Default properties (in UI? when not specified in boto3?) and Limits 
- Superlative hell: Primary/Best/Easiest
- All True, n + 1, n+2, n+3 are true on a "(Select n)" 
- Weasel words: optimal, primary, default, principal, main benefit, etc

## Explanations
- They need a format, e.g. "..., therefore wrong. ..., therefore correct."