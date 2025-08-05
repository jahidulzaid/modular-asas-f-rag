# %% [markdown]
# # Setup Environment

# %%
import os
import json
from typing import Type
from typing import Any
from inspect import isclass
import random

import dspy
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
# from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
# from dspy.teleprompt import BayesianSignatureOptimizer
from sklearn.model_selection import train_test_split

random.seed(1)



# %%
import langwatch

langwatch.login()

# %%
TASK_MODEL_NAME = "mistral:7b"
PROMPT_MODEL_NAME = "llama3:70b"

# %%
prompt_llm = dspy.OllamaLocal(model=PROMPT_MODEL_NAME, temperature=0.0, timeout_s=240, max_tokens=8000)
task_llm = dspy.OllamaLocal(model=TASK_MODEL_NAME, temperature=0.0,timeout_s=240, max_tokens=8000)
dspy.configure(experimental=True)
dspy.settings.configure(lm=prompt_llm)


# %% [markdown]
# # ASAS-F-Opt Program

# %%
# Define optimization metric

def validate_label(example, pred, trace=None):
    return example.feedback.label.lower() == pred.feedback.label.lower()

class Answer(BaseModel):
    question: str = Field(description="The question posed to the student")
    student_answer: str = Field(description="The student's written answer")
    reference_answer: str = Field(desc="The reference material for the question")


class Feedback(BaseModel):
    label: str = Field(description="Either correct, partially correct, or incorrect.")
    numeric_score: float = Field(description="Grading score out of 1")
    explanation: str = Field(description="Rationale behind score and label")
        
class StudentAnswerScoring(dspy.Signature):
    """
    Score a student's answer against a reference, providing a label,
    numerical score, and reasoning.
    Scoring criteria: 
    -Correct: The student's answer demonstrates a clear understanding of the core concept, with key points accurately addressed. Minor errors are acceptable
    - Partially Correct: The student's answer shows some understanding of the core concept but misses key points or contains notable inaccuracies.
    - Incorrect: The student's answer fails to demonstrate an understanding of the core concept or is largely inaccurate.
    """
    
    answer: Answer = dspy.InputField()
    feedback: Feedback = dspy.OutputField()


scorer = dspy.TypedPredictor(StudentAnswerScoring, max_retries=8)

# %% [markdown]
# # Load Datasets

# %%
import pandas as pd
df_saf = pd.read_csv('data/train.csv')
df_saf = df_saf.fillna('')
df_saf['score'] = df_saf['score'].astype(float)

saf_trainset = []
for index, row in df_saf.iterrows():
        example = dspy.Example(answer=Answer(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), feedback=Feedback(
                        label=row['label'], numeric_score=row['score'], explanation=row['feedback'])).with_inputs('answer')
        saf_trainset.append(example)

random.shuffle(saf_trainset)
trainset, valset = saf_trainset[:500], saf_trainset[500:1000]

# %%
# Unseen Answers
df_ua = pd.read_csv('data/ua.csv')
df_ua = df_ua.fillna('')

columns = ['question', 'student', 'reference', 'label', 'score', 'feedback']
data_ua = df_ua[columns]
ua_set = []
for index, row in data_ua.iterrows():
    example = Answer(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), Feedback(
                          label=row['label'], numeric_score=row['score'], explanation=row['feedback'])

    ua_set.append(example)

# Unseen Questions
df_uq = pd.read_csv('data/uq.csv')
df_uq = df_uq.fillna('')

columns = ['question', 'student', 'reference', 'label', 'score', 'feedback']
data_uq = df_uq[columns]
uq_set = []
for index, row in data_uq.iterrows():
    example = Answer(question=row['question'], student_answer=row['student'], reference_answer=row['reference']), Feedback(
                          label=row['label'], numeric_score=row['score'], explanation=row['feedback'])

    uq_set.append(example)

# %% [markdown]
# # Optimization with MIPROv2

# %%
from dspy.teleprompt import MIPROv2

teleprompter = MIPROv2(prompt_model=prompt_llm, task_model=task_llm, metric=validate_label, num_candidates=3, init_temperature=0.0, verbose=True)
langwatch.dspy.init(experiment="saf-miprov2", optimizer=optimizer)

kwargs = dict(display_progress=True, display_table=0, num_threads=16)

compiled_program = teleprompter.compile(scorer, trainset=trainset, valset=valset, num_batches=20, max_bootstrapped_demos=1, max_labeled_demos=3, eval_kwargs=kwargs)

# %%
compiled_program.save("outputs/asas-f-opt-compiled.json")

# %%
task_llm = dspy.OllamaLocal(model=TASK_MODEL_NAME, temperature=0.0,timeout_s=240)

dspy.settings.configure(lm=task_llm)

# %% [markdown]
# # Generate Outputs for Unseen Answers 

# %%
generated_outputs_ua = []
correct_count = 0
total_count = 0

for ua in tqdm(ua_set, desc="Processing"):
    try:
        pred = compiled_program(answer=Answer(question=ua[0].question, student_answer=ua[0].student_answer, reference_answer=ua[0].reference_answer))
        pred = pred.feedback
    except Exception as e:
        print(f"Error processing UA: {e}")
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
with open(f'outputs/ua_opt_{TASK_MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_ua, json_file, indent=2)

generated_df_ua = pd.DataFrame(generated_outputs_ua)
generated_df_ua.to_csv(f'outputs/ua_opt_{TASK_MODEL_NAME}.csv', index=False)

# %% [markdown]
# # Generate Outputs for Unseen Questions

# %%
generated_outputs_uq = []
correct_count = 0
total_count = 0

for uq in tqdm(uq_set, desc="Processing"):
    try:
        pred = compiled_program(answer=Answer(question=uq[0].question, student_answer=uq[0].student_answer, reference_answer=uq[0].reference_answer))
        pred = pred.feedback
    except Exception as e:
        print(f"Error processing UQ: {e}")
        pred = type('pred', (object,), {'label': "error", 'numeric_score': 'error', 'explanation': 'error'})

    if pred.label.lower() == uq[1].label.lower():
        correct_count += 1
    total_count += 1

    generated_outputs_ua.append({
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
with open(f'outputs/uq_opt_{TASK_MODEL_NAME}.json', 'w') as json_file:
    json.dump(generated_outputs_uq, json_file, indent=2)

generated_df_uq = pd.DataFrame(generated_outputs_uq)
generated_df_uq.to_csv(f'outputs/uq_opt_{TASK_MODEL_NAME}.csv', index=False)


