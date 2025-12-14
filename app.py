from flask import Flask, render_template, request, jsonify, session, send_file
import random
import json
import torch
from datetime import datetime
import secrets
import logging
import os
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_RIGHT

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

# Load the model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('cricket_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

# Create logs directory if it doesn't exist
if not os.path.exists('chat_logs'):
    os.makedirs('chat_logs')

# Store chat history in session
def add_to_chat_history(session_id, sender, message):
    """Add a message to the chat history"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append({
        'sender': sender,
        'message': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    session.modified = True

def get_session_logger(session_id):
    """Create or get logger for a specific session"""
    logger = logging.getLogger(session_id)
    
    # Only add handler if it doesn't exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        log_file = f'chat_logs/session_{session_id}.log'
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_response(message, session_id):
    """Process user message and return bot response"""
    logger = get_session_logger(session_id)
    start_time = time.time()
    
    # Log user input
    logger.info(f"User Input: {message}")
    
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    processing_time = time.time() - start_time
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                logger.info(f"Bot Response: {response} (tag: {tag}, confidence: {prob.item():.4f})")
                logger.info(f"Processing time: {processing_time:.4f} seconds")
                return response, prob.item()
    
    response = "I do not understand..."
    logger.info(f"Bot Response: {response} (confidence: {prob.item():.4f})")
    logger.info(f"Processing time: {processing_time:.4f} seconds")
    return response, prob.item()

@app.route('/')
def home():
    """Render the main chat interface"""
    # Create a new session ID if it doesn't exist
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = get_session_logger(session['session_id'])
        logger.info("==== New Chat Session Started ====")
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        user_message = request.json.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = get_session_logger(session['session_id'])
            logger.info("==== New Chat Session Started ====")
        
        # Add user message to chat history
        add_to_chat_history(session['session_id'], 'user', user_message)
        
        # Get bot response with logging
        bot_response, confidence = get_response(user_message, session['session_id'])
        
        # Add bot response to chat history
        add_to_chat_history(session['session_id'], 'bot', bot_response)
        
        return jsonify({
            'response': bot_response,
            'confidence': round(confidence, 4),
            'bot_name': bot_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_chat', methods=['GET'])
def download_chat():
    """Download the current chat session as PDF"""
    try:
        if 'chat_history' not in session or len(session['chat_history']) == 0:
            return jsonify({'error': 'No chat history found'}), 404
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles for user and bot messages
        user_style = ParagraphStyle(
            'UserStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#4a5568',
            alignment=TA_RIGHT,
            spaceAfter=6,
            leftIndent=100,
            rightIndent=0,
        )
        
        bot_style = ParagraphStyle(
            'BotStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2d3748',
            alignment=TA_LEFT,
            spaceAfter=6,
            leftIndent=0,
            rightIndent=100,
        )
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=18,
            textColor='#667eea',
            spaceAfter=20,
            alignment=TA_LEFT,
        )
        
        # Add title
        title = Paragraph("🏏 Cricket Chatbot - Conversation History", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add chat messages
        for entry in session['chat_history']:
            sender = entry['sender']
            message = entry['message']
            
            if sender == 'user':
                label = Paragraph("<b>You:</b>", user_style)
                elements.append(label)
                msg = Paragraph(message, user_style)
                elements.append(msg)
            else:
                label = Paragraph("<b>Bot:</b>", bot_style)
                elements.append(label)
                msg = Paragraph(message, bot_style)
                elements.append(msg)
            
            elements.append(Spacer(1, 0.15*inch))
        
        # Build PDF
        doc.build(elements)
        
        # Move buffer position to beginning
        buffer.seek(0)
        
        # Log the download action
        if 'session_id' in session:
            logger = get_session_logger(session['session_id'])
            logger.info("==== Chat PDF Downloaded ====")
        
        session_id = session.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'chat_history_{session_id}.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    """End the current chat session"""
    try:
        if 'session_id' in session:
            logger = get_session_logger(session['session_id'])
            logger.info("==== Chat Session Ended ====")
            
            # Remove handlers to release the log file
            logger.handlers.clear()
            
            session_id = session['session_id']
            session.clear()
            
            return jsonify({'message': 'Session ended', 'session_id': session_id})
        
        return jsonify({'message': 'No active session'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the chat session"""
    try:
        if 'session_id' in session:
            logger = get_session_logger(session['session_id'])
            logger.info("==== Chat Session Reset ====")
            logger.handlers.clear()
        
        session.clear()
        return jsonify({'message': 'Chat session reset'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)