from flask import Flask, send_file, jsonify
import sqlite3
import os 
import time
import logging
import threading
import mimetypes
import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
from transformers import MobileBertTokenizer

app = Flask(__name__)
CONTENT_DIR = "./content"
DB_FILE = "./db/lms.db"
MODEL_PATH_1 = "./models/mobilebert-litert/1.tflite"
MODEL_PATH_2 = "./models/mobilebert-litert/tf_model.h5"
TOKENIZER_PATH = "./models/mobilebert-litert"

#logging 
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lms.log'),
        logging.StreamHandler()
    ]
)

#load mobilebert tokenizer and tflite model
try:
    tokenizer = MobileBertTokenizer.from_pretrained(TOKENIZER_PATH)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH_1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("MobileBERT LiteRT model and tokenizer loaded successfullly")
    logging.info("Input details: %s", input_details)
    logging.info("Output details: %s", output_details)
except Exception as e:
    logging.error("Failed to load MobileBERT LiteRT model or tokenizer: %s", e)
    raise

#thread-local storage for SQLite connections
thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, 'connection'):
        try:
            thread_local.connection = sqlite3.connect(DB_FILE, check_same_thread=False)
            thread_local.connection.row_factory = sqlite3.Row
            cursor = thread_local.connection.cursor()
            cursor.execute(''' CREATE TABLE IF NOT EXISTS progress
                           (student_id TEXT, module_id TEXT, score INTEGER, timestamp INTEGER) ''')
            thread_local.connection.commit()
            logging.info(f"SQLite Connection initialized for thread {threading.get_ident()}")
        except Exception as e:
            logging.error(f"Failed to initialize SQLite for thread {threading.get_ident()}:{e}")
            raise
    return thread_local.connection

def get_mobilebert_recommendation(score):
    """Generate recommendation using MobileBERT LiteRT based on quiz score."""
    try:
        # Create a QA-style prompt with context
        question = f"What should a student who scored {score}% on a quiz do next?"
        context = (
            "For scores below 60 percent, review basic concepts to build a stronger foundation. "
            "For scores between 60 percent and 80 percent, practice more to improve understanding. "
            "For scores above 80 percent, advance to more challenging topics to deepen your skills."
        )
        inputs = tokenizer(
            question,
            context,
            return_tensors="tf",
            max_length=384,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        input_ids = inputs["input_ids"].numpy().astype(np.int32)
        input_mask = inputs["attention_mask"].numpy().astype(np.int32)
        segment_ids = inputs["token_type_ids"].numpy().astype(np.int32)

        # Set input tensors
        for input_detail in input_details:
            if input_detail["name"] == "input_ids":
                interpreter.set_tensor(input_detail["index"], input_ids)
            elif input_detail["name"] == "input_mask":
                interpreter.set_tensor(input_detail["index"], input_mask)
            elif input_detail["name"] == "segment_ids":
                interpreter.set_tensor(input_detail["index"], segment_ids)
            else:
                logging.warning("Skipping unexpected input: %s", input_detail["name"])

        interpreter.invoke()

        # Get output (start_logits and end_logits)
        start_logits = interpreter.get_tensor(output_details[0]["index"])
        end_logits = interpreter.get_tensor(output_details[1]["index"])
        
        # Extract answer span
        start_idx = np.argmax(start_logits[0])
        end_idx = np.argmax(end_logits[0])
        answer_tokens = input_ids[0][start_idx:end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        # Map score to expected recommendation for validation
        if score < 60:
            expected = "review basic concepts to build a stronger foundation"
        elif score < 80:
            expected = "practice more to improve understanding"
        else:
            expected = "advance to more challenging topics to deepen your skills"
        
        # Validate answer; use expected recommendation if answer is incomplete or incorrect
        if not answer or expected.lower() not in answer.lower() or len(answer.split()) < 5:
            answer = expected.capitalize() + "."
        
        logging.info("Generated recommendation for score %s: %s", score, answer)
        return answer
    except Exception as e:
        logging.error("Error generating recommendation: %s", e)
        # Fallback to rule-based recommendation
        if score < 60:
            return "Review basic concepts to build a stronger foundation."
        elif score < 80:
            return "Practice more to improve understanding."
        else:
            return "Advance to more challenging topics to deepen your skills."

@app.route("/")
def index():
    try:
        if os.path.exists("content/index.html"):
            logging.info("Serving index.html")
            return send_file("content/index.html")
        logging.error("index.html not found")
        return jsonify({"error":"Main page not found"}), 404
    except Exception as e:
        logging.error("Error serving index.html:%s", e)
        return jsonify({"error":"Server error"}), 500

@app.route("/content/<path:filename>")
def serve_content(filename):
    try:
        file_path = os.path.join(CONTENT_DIR, filename)
        if not os.path.exists(file_path):
            logging.warning("File not found: %s", file_path)
            return jsonify({"error":"File not found"}), 404
        mime_type,_=mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type='application/octet-stream'
        logging.info("Serving file: %s (MIME:%s)", file_path, mime_type)
        return send_file(file_path, mimetype=mime_type)
    except Exception as e:
        logging.error("Error serving content %s", e)
        return jsonify({"error":"Server error"}), 500


@app.route("/quiz/<student_id>/<score>")
def submit_quiz(student_id, score):
    try:
        if not student_id or not student_id.strip():
            logging.warning("Invalid student_id %s",student_id)
            return jsonify({"error":"Invalid student ID"}), 400
        score=int(score)
        if score < 0 or score > 100:
            logging.warning("Invalid score: %s", score)
            return jsonify({"error": "Invalid Score"}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO progress (student_id, module_id, score, timestamp) VALUES(?,?,?,?)",
                       (student_id, "math_quiz_1", score, int(time.time())))
        conn.commit()
        logging.info("Quiz submitted: student_id=%s, score=%s, thread=%s", student_id, score, threading.get_ident())
        recommendation = get_mobilebert_recommendation(score)
        return jsonify({"score":score, "recommendation":recommendation})
    except ValueError:
        logging.warning("Invalid score format: %s", score)
        return jsonify({"error": "Invalid score format"}), 400
    except sqlite3.Error as e:
        logging.error("SQLite error in submit_quiz: %s", e)
        return jsonify({"error": "Databse error: " + str(e)}), 500
    except Exception as e:
        logging.error("Unexpected error in submit_quiz: %s", e)
        return jsonify({"error": "Server error: " + str(e)}), 500

@app.teardown_appcontext
def close_connection(exception):
    """Close thread-local SQLite connection at the end of each request."""
    if hasattr(thread_local,'connection'):
        thread_local.connection.close()
        del thread_local.connection
        logging.info(f"Closed SQLite connection for thread {threading.get_ident()}")

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8080, debug=False)
    except Exception as e:
        logging.error("Failed to start Flask server: %s", e)
