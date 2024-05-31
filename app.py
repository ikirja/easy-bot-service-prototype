from flask import Flask, request, jsonify
from sentence_similarity import compare
from text_generation import get_answer

app = Flask('main')


@app.post('/sentence-similarity')
def sentence_similarity():
    question = request.json['question']
    variants = request.json['variants']
    similarity_result = compare(question, variants)

    return jsonify(similarity_result)


@app.post('/text-generation')
def text_generation():
    system_prompt = request.json['systemPrompt']
    prompt = request.json['prompt']
    answer = get_answer(system_prompt, prompt)

    return jsonify({'answer': answer})


app.run(port=4000)
