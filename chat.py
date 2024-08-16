import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'sk-proj-wBhxVeSmc5c9wq0MccFNT3BlbkFJPnPgz351rUnyoyLziIRu'

# Manually set the correct column names
columns = ['Question', 'Answers', 'Recommendation', 'LINK', 'KeyWords']

# Read the TSV file, assuming it's tab-separated, and assign column names
data = pd.read_csv('renew-u.tsv', sep='\t', header=None, names=columns)

# Strip any extra spaces from column names (just in case)
data.columns = data.columns.str.strip()

# Combine context for embeddings by merging 'Answers' and 'Recommendation' columns
data['combined_context'] = data['Answers'] + " " + data['Recommendation']

# Load a smaller model and specify that it should run on the CPU
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')

# Generate embeddings using the CPU
embeddings = model.encode(data['combined_context'].tolist(), device='cpu')

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Function to generate a response using ChatGPT (gpt-3.5-turbo)
def generate_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful wellness companion."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to query the chatbot and get a ChatGPT-generated response
def query_chatbot(query):
    query_embedding = model.encode([query], device='cpu')
    D, I = index.search(query_embedding, k=1)  # Search for the closest match
    result = data.iloc[I[0][0]]
    
    # Create a prompt for ChatGPT to rephrase the recommendation
    prompt = (
        f"You asked: {query}\n\n"
        f"Here is a situation related to your query:\n\n"
        f"{result['Answers']}\n\n"
        f"Please rephrase the following recommendation to make it more engaging: "
        f"{result['Recommendation']}\n\n"
        f"You can also watch this video for more guidance: {result['LINK']}"
    )
    
    # Generate the ChatGPT response
    chatgpt_response = generate_chatgpt_response(prompt)
    
    return {
        'chatgpt_response': chatgpt_response,
        'youtube_link': result['LINK'],
        'recommendation': result['Recommendation']
    }

# Chatbot loop
def start_chatbot():
    print("Welcome to the Wellness Companion Chatbot! Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Take care!")
            break
        
        # Get the chatbot response
        response = query_chatbot(user_input)
        
        # Print the chatbot response
        print(f"Chatbot: {response['chatgpt_response']}")
        #print(f"Watch this video for more guidance: {response['youtube_link']}\n")

# Start the chatbot
start_chatbot()
