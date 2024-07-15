from flask import Flask, request, jsonify
import flask_cors
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from model import recog
from haircolormodel import main

load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyCYeE0wT2nMhNV5aIwp1tpLrdKc37GRizM"
app = Flask(__name__)
flask_cors.CORS(app)

def description_generation(skin, eyes, hair):
    query = '''skin: {skin}
hair: {hair}
eyes: {eyes}
which season am i according to colour analysis. give me a list of colours that would suit me according to my colour palette'''
    query.format(skin, hair, eyes)
    response = model.generate_content(query)
    response.__format__
    re = response.text
    re = re[4:]
    re = re[:-3]
    json_data = json.loads(re)
    return jsonify(json_data)

def colour_generation(skin, eyes, hair):
    query = '''skin: {skin}
hair: {hair}
eyes: {eyes}
which season am i according to colour analysis. give me a list of colours that would suit me according to my colour palette. the colours should be in the form of a python array'''
    query.format(skin, hair, eyes)
    response = model.generate_content(query)
    response.__format__
    re = response.text
    re = re[4:]
    re = re[:-3]
    json_data = json.loads(re)
    return jsonify(json_data)

def get_colours(img_loc):
    skin, eyes = recog(img_loc)
    hair = main(img_loc)
    return skin, eyes, hair



model = genai.GenerativeModel('gemini-pro')

@app.route("/reccomend", methods=["POST"])
def reccomend():
    request_data = request.get_json()
    img_loc = request_data["message"]
    print(img_loc)
    skin, eyes, hair = get_colours(img_loc)
    des = description_generation(skin, eyes, hair)
    clrs = colour_generation(skin, eyes, hair)
    # print()
    message = {"description": des["output_text"], "colours": clrs["output_text"] }
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
