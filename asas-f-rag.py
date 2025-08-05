# %% [markdown]
# # Setup the environment
# 

# %%
import os
import json
from typing import Type
from typing import Any
from inspect import isclass
import random

import dspy
from dsp import passages2text
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy.teleprompt import BayesianSignatureOptimizer
from sklearn.model_selection import train_test_split

# %%
MODEL_NAME = "mistral:7b"
NUM_EXAMPLES = 3

# %%
llm = dspy.OllamaLocal(model=MODEL_NAME, max_tokens=8192, temperature=0.0)
colbertv2_saf = dspy.ColBERTv2(url='http://127.0.0.1:8890/api/search')
dspy.settings.configure(lm=llm)
dspy.settings.configure(rm=colbertv2_saf)

# %% [markdown]
# # ASAS-F-RAG Program

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
    reference_answer: str = Field(description="The reference material for the question")
    student_answer: str = Field(description="The student's written answer")

class Output(BaseModel):
    label: str = Field(description="Either correct, partially correct, or incorrect.")
    numeric_score: float = Field(description="Grading score out of 1")
    feedback: str = Field(description="Rationale behind score and label")
        
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

class FeedbackGeneratorRAG(dspy.FunctionalModule):
    def __init__(self, num_passages=NUM_EXAMPLES):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        
    @dspy.predictor
    def output(self, examples, input: Input) -> Output:
        """
    Score a student's answer against a reference, providing a label,
    numerical score, and feedback.
    Scoring criteria: 
    - Correct: The student's answer demonstrates a clear understanding of the core concept, with key points accurately addressed. Minor errors are acceptable.
    - Partially Correct: The student's answer shows some understanding of the core concept but misses key points or contains notable inaccuracies.
    - Incorrect: The student's answer fails to demonstrate an understanding of the core concept or is largely inaccurate.
        """  
        pass

    def forward(self, question, reference_answer, student_answer):  
        retrieved = self.retrieve(student_answer).passages
        similar_scored_examples = passages2text(retrieved)
        input_data = Input(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer
        )

        scoring = self.output(examples=similar_scored_examples, input=input_data)
        return dspy.Prediction(label=scoring.label, numeric_score=scoring.numeric_score, feedback=scoring.feedback)

scorer = FeedbackGeneratorRAG()

# %% [markdown]
# # ASAS-F-RAG with no type constraints

# %%
class AutomaticStudentAnswerScoring(dspy.Signature):
    """
    Score a student's answer against a reference answer by providing a label, numerical score, and reasoning.
    Use additional context from similar scored examples to understand grading patterns and the assessment process.
    These examples are for reference only and are meant to guide the scoring by illustrating how similar answers have been evaluated previously, not for direct comparison or labeling.
    """
    question = dspy.InputField(desc="The question posed to the student.")
    reference_answer = dspy.InputField(desc="The reference answer.")

    similar_scored_examples = dspy.InputField(
        desc="A list of similar answered responses with their scores and feedback. Use for context to understand grading patterns.",
    )
    
    student_answer = dspy.InputField(desc="The answer to score.")
    
    label = dspy.OutputField(desc="A label either: correct, partially correct, or incorrect.")
    numeric_score = dspy.OutputField(desc="Score out of 1")
    feedback = dspy.OutputField(desc="Rationale behind scores and label.")

from dsp import passages2text


class AutomaticFeedbackGeneratorRAG(dspy.Module):
    def __init__(self, num_passages=NUM_EXAMPLES):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.score_student_answer = dspy.Predict(AutomaticStudentAnswerScoring)

    def forward(self, question, reference_answer, student_answer):        
        similar_scored_examples = passages2text(self.retrieve(student_answer).passages)
        scoring = self.score_student_answer(question=question, reference_answer=reference_answer, student_answer=student_answer, similar_scored_examples=similar_scored_examples)
        return dspy.Prediction(label=scoring.label, numeric_score = scoring.numeric_score, feedback=scoring.feedback)

automatic_scorer = AutomaticFeedbackGeneratorRAG()

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
                          label=row['label'], numeric_score=row['score'], feedback=row['feedback'])

    ua_set.append(example)

# Unseen Questions
df_uq = pd.read_csv('data/uq.csv')
df_uq = df_uq.fillna('')

columns = ['question', 'student', 'reference', 'label', 'score', 'feedback']
data_uq = df_uq[columns]
uq_set = []
for index, row in data_uq.iterrows():
    example = Input(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), Output(
                          label=row['label'], numeric_score=row['score'], feedback=row['feedback'])

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
            pred = type('pred', (object,), {'label': "error", 'numeric_score': 'error', 'feedback': 'error'})

    if pred.label.lower() == ua[1].label.lower():
        correct_count += 1
    total_count += 1

    generated_outputs_ua.append({
        'question': ua[0].question,
        'student': ua[0].student_answer,
        'reference': ua[0].reference_answer,
        'label': ua[1].label,
        'score': ua[1].numeric_score,
        'feedback': ua[1].feedback,
        'pred_label': pred.label,
        'pred_score': pred.numeric_score,
        'pred_feedback': pred.feedback
    })

    accuracy = correct_count / total_count if total_count > 0 else 0
    tqdm.write(f"Accuracy: {accuracy:.2%}")


# %%
with open(f'outputs/ua_rag_{NUM_EXAMPLES}_{MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_ua, json_file, indent=2)

generated_df_ua = pd.DataFrame(generated_outputs_ua)
generated_df_ua.to_csv(f'outputs/ua_rag_{NUM_EXAMPLES}_{MODEL_NAME}.csv', index=False)

# %% [markdown]
# # Generate Outputs for Unseen Questions

# %%

generated_outputs_uq = []
correct_count = 0
total_count = 0

for uq in tqdm(ua_set, desc="Processing"):
    try:
        pred = scorer(input=Input(question=uq[0].question, student_answer=uq[0].student_answer, reference_answer=uq[0].reference_answer))
        pred = pred.output
    except Exception as e:
        try:
            pred = automatic_scorer(question=uq[0].question, student_answer=uq[0].student_answer, reference_answer=uq[0].reference_answer)
        except Exception as e2:
            print(f"Error processing UQ with both pipelines: {e2}")
            pred = type('pred', (object,), {'label': "error", 'numeric_score': 'error', 'feedback': 'error'})

    if pred.label.lower() == uq[1].label.lower():
        correct_count += 1
    total_count += 1

    generated_outputs_uq.append({
        'question': uq[0].question,
        'student': uq[0].student_answer,
        'reference': uq[0].reference_answer,
        'label': uq[1].label,
        'score': uq[1].numeric_score,
        'feedback': uq[1].feedback,
        'pred_label': pred.label,
        'pred_score': pred.numeric_score,
        'pred_feedback': pred.feedback
    })

    accuracy = correct_count / total_count if total_count > 0 else 0
    tqdm.write(f"Accuracy: {accuracy:.2%}")


# %%
with open(f'outputs/uq_rag_{NUM_EXAMPLES}_{MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_uq, json_file, indent=2)

generated_df_uq = pd.DataFrame(generated_outputs_uq)
generated_df_uq.to_csv(f'outputs/uq_rag_{NUM_EXAMPLES}_{MODEL_NAME}.csv', index=False)


