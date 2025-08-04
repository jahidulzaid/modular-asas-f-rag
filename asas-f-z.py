# %% [markdown]
# # Setup the environment
# 

# %%



# %%
import os
import json
from typing import Type
from typing import Any
from inspect import isclass

import dspy
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy.teleprompt import BayesianSignatureOptimizer
from sklearn.model_selection import train_test_split




# %%
MODEL_NAME = "mistral:7b"

# %%
llm = dspy.OllamaLocal(model=MODEL_NAME, max_tokens=8192, temperature=0.0)
dspy.settings.configure(lm=llm)

# %% [markdown]
# # ASAS-F-Z Program

# %%
def get_min_length(model: Type[BaseModel]):
    min_length = 0
    for key, field in model.model_fields.items():
        min_length += len(key)
        if not isclass(field.annotation):
            if issubclass(field.annotation, BaseModel):
                min_length += get_min_length(field.annotation)
    return min_length

class Input(BaseModel):
    question: str = Field(description="The question posed to the student")
    student_answer: str = Field(description="The student's written answer")
    reference_answer: str = Field(desc="The reference material for the question")

class Output(BaseModel):
    label: str = Field(description="Either correct, partially correct, or incorrect.")
    numeric_score: float = Field(description="Grading score out of 1")
    explanation: str = Field(description="Rationale behind score and label")
        
    @classmethod
    def model_validate_json(
        cls,
        json_data: str,
        *,
        strict: bool | None = None,
        context: dict[str, Any] | None = None
    ) -> "Output":
        __tracebackhide__ = True
        try:
            return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)
        except ValidationError:
            min_length = get_min_length(cls)
            for substring_length in range(len(json_data), min_length-1, -1):
                for start in range(len(json_data)-substring_length+1):
                    substring = json_data[start:start+substring_length]
                    try:
                        res = cls.__pydantic_validator__.validate_json(substring, strict=strict, context=context)
                        return res
                    except ValidationError:
                        pass
        raise ValueError("Could not find valid json")

class StudentAnswerScoring(dspy.Signature):
    """
    Score a student's answer against a reference, providing a label,
    numerical score, and reasoning.
    Scoring criteria: 
    -Correct: The student's answer demonstrates a clear understanding of the core concept, with key points accurately addressed. Minor errors are acceptable
    - Partially Correct: The student's answer shows some understanding of the core concept but misses key points or contains notable inaccuracies.
    - Incorrect: The student's answer fails to demonstrate an understanding of the core concept or is largely inaccurate.
    """
    
    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()


scorer = dspy.TypedPredictor(StudentAnswerScoring)

# %% [markdown]
# ## ASAS-F-Z with no type constraints

# %%
class AutomaticStudentAnswerScoring(dspy.Signature):
    """
    Score a student's answer against a reference answer, providing a label,
    numerical score, and reasoning.
    """
    question = dspy.InputField(desc="The question posed to the student.")
    student_answer = dspy.InputField(desc="The student's written answer.")
    reference_answer = dspy.InputField(desc="The reference or correct answer to the question.")

    label = dspy.OutputField(desc="Correct, partially correct, or incorrect.")
    numeric_score = dspy.OutputField(desc="Accuracy score out of 1.")
    explanation = dspy.OutputField(desc="Rationale behind score and label.")
    
class FeedbackGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.score_student_answer = dspy.Predict(AutomaticStudentAnswerScoring)

    def forward(self, question, reference_answer, student_answer):
        scoring = self.score_student_answer(question=question, reference_answer=reference_answer, student_answer=student_answer)
        return dspy.Prediction(label=scoring.label, numeric_score = scoring.numeric_score, explanation=scoring.explanation)

automatic_scorer = FeedbackGenerator()

# %% [markdown]
# # Load the Dataset

# %%
# Unseen Answers
df_ua = pd.read_csv('data/ua.csv')
df_ua = df_ua.fillna('')

columns = ['question', 'student', 'reference', 'label', 'score', 'feedback']
data_ua = df_ua[columns]
ua_set = []
for index, row in data_ua.iterrows():
    example = Input(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), Output(
                          label=row['label'], numeric_score=row['score'], explanation=row['feedback'])

    ua_set.append(example)

# Unseen Questions
df_uq = pd.read_csv('data/uq.csv')
df_uq = df_uq.fillna('')

columns = ['question', 'student', 'reference', 'label', 'score', 'feedback']
data_uq = df_uq[columns]
uq_set = []
for index, row in data_uq.iterrows():
    example = Input(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), Output(
                          label=row['label'], numeric_score=row['score'], explanation=row['feedback'])

    uq_set.append(example)

# %% [markdown]
# # Generate Outputs for Unseen Answers 

# %%

generated_outputs_ua = []
correct_count = 0
total_count = 0

for ua in tqdm(ua_set, desc="Processing"):
    try:
        pred = scorer(input=Input(question=ua[0].question, student_answer=ua[0].student_answer, reference_answer=ua[0].reference_answer))
        pred = pred.output
    except Exception as e:
        try:
            pred = automatic_scorer(question=ua[0].question, student_answer=ua[0].student_answer, reference_answer=ua[0].reference_answer)
        except Exception as e2:
            print(f"Error processing UA with both pipelines: {e2}")
            pred = type('pred', (object,), {'label': "error", 'numeric_score': 'error', 'explanation': 'error'})

    if pred.label.lower() == ua[1].label.lower():
        correct_count += 1
    total_count += 1

    generated_outputs_ua.append({
        'question': ua[0].question,
        'student': ua[0].student_answer,
        'reference': ua[0].reference_answer,
        'label': ua[1].label,
        'score': ua[1].numeric_score,
        'feedback': ua[1].explanation,
        'pred_label': pred.label,
        'pred_score': pred.numeric_score,
        'pred_feedback': pred.explanation
    })

    accuracy = correct_count / total_count if total_count > 0 else 0
    tqdm.write(f"Accuracy: {accuracy:.2%}")


# %%
with open(f'outputs/ua_zeroshot_{MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_ua, json_file, indent=2)

generated_df_ua = pd.DataFrame(generated_outputs_ua)
generated_df_ua.to_csv(f'outputs/ua_zeroshot_{MODEL_NAME}.csv', index=False)

# %% [markdown]
# # Generate Outputs for Unseen Questions

# %%
generated_outputs_uq = []
correct_count = 0
total_count = 0

for uq in tqdm(uq_set, desc="Processing"):
    try:
        pred = scorer(input=Input(question=uq[0].question, student_answer=uq[0].student_answer, reference_answer=uq[0].reference_answer))
        pred = pred.output
    except Exception as e:
        try:
            pred = automatic_scorer(question=uq[0].question, student_answer=uq[0].student_answer, reference_answer=uq[0].reference_answer)
        except Exception as e2:
            print(f"Error processing UQ with both pipelines: {e2}")
            pred = type('pred', (object,), {'label': "error", 'numeric_score': 'error', 'explanation': 'error'})

    if pred.label.lower() == uq[1].label.lower():
        correct_count += 1
    total_count += 1

    generated_outputs_uq.append({
        'question': uq[0].question,
        'student': uq[0].student_answer,
        'reference': uq[0].reference_answer,
        'label': uq[1].label,
        'score': uq[1].numeric_score,
        'feedback': uq[1].explanation,
        'pred_label': pred.label,
        'pred_score': pred.numeric_score,
        'pred_feedback': pred.explanation
    })

    accuracy = correct_count / total_count if total_count > 0 else 0
    tqdm.write(f"Accuracy: {accuracy:.2%}")


# %%
with open(f'outputs/uq_zeroshot_{MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_uq, json_file, indent=2)

generated_df_uq = pd.DataFrame(generated_outputs_uq)
generated_df_uq.to_csv(f'outputs/uq_zeroshot_{MODEL_NAME}.csv', index=False)


