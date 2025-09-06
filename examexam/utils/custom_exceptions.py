class ExamExamError(Exception):
    """ExamExam threw this error"""


class ExamExamValueError(ExamExamError):
    """ExamExam does not like that value"""


class ExamExamTypeError(ExamExamError):
    """ExamExam does not like that"""
