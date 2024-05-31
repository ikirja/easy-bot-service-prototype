from llama_cpp import Llama
from numpy import (array, dot)
from numpy.linalg import norm

llm = Llama(
    model_path='models/all-MiniLM-L6-v2.Q4_0.gguf',
    embedding=True
)


def compare(question, variants):
    result = []

    embedded_question = llm.embed(question)
    question_vector_array = array(embedded_question)

    for variant in variants:
        embedded_variant = llm.embed(variant)
        variant_vector_array = array(embedded_variant)

        cosine = dot(question_vector_array, variant_vector_array) / norm(question_vector_array) / norm(variant_vector_array)

        result.append(cosine)

    return result

