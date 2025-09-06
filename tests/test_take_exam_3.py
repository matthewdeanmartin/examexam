import pytest

from examexam.take_exam import take_exam_machine  # or the module name you use


@pytest.fixture
def exam_file(tmp_path):
    """Create a temporary TOML file with two exam questions."""
    content = """
    [[questions]]
    id = "11111111-1111-4111-8111-111111111111"
    question = "Which AWS service lets you run SQL directly on data in S3? (Select 1)"
    
    
    [[questions.options]]
    text = "Amazon Athena"
    explanation = "Athena is serverless and queries S3 with SQL. Correct."
    is_correct = true
    
    
    [[questions.options]]
    text = "Amazon RDS"
    explanation = "RDS is for managed relational databases. Incorrect."
    is_correct = false
    
    
    [[questions.options]]
    text = "Amazon Redshift"
    explanation = "Data warehouse service; not direct ad‑hoc on raw S3 objects. Incorrect."
    is_correct = false
    
    
    [[questions]]
    id = "22222222-2222-4222-8222-222222222222"
    question = "Choose two highly durable AWS storage classes. (Select 2)"
    
    
    [[questions.options]]
    text = "S3 Standard"
    explanation = "Designed for 11 nines of durability. Correct."
    is_correct = true
    
    
    [[questions.options]]
    text = "S3 Glacier Deep Archive"
    explanation = "Also 11 nines durability for archival data. Correct."
    is_correct = true
    
    
    [[questions.options]]
    text = "Instance Store"
    explanation = "Ephemeral, not durable. Incorrect."
    is_correct = false
    
    
    [[questions.options]]
    text = "EFS Infrequent Access"
    explanation = "Durable, but not typically cited at the same durability level as S3 classes; incorrect here."
    is_correct = false
    """
    exam_file = tmp_path / "athena_sample.toml"
    exam_file.write_text(content.strip(), encoding="utf-8")
    return exam_file


def test_exam_machine_oracle(exam_file):
    result = take_exam_machine(str(exam_file), strategy="oracle", seed=123, quiet=True)
    assert result["total"] == 2
    # In oracle mode we should be perfect if TOML marks correct answers
    assert result["score"] == result["total"]
    assert result["percent"] == 100.0
