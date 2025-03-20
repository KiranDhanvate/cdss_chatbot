# RAG-Enhanced Clinical Decision Support System

This project implements a Retrieval-Augmented Generation (RAG) model in Django with a simple HTML frontend. The system provides clinical decision support by analyzing medical queries, retrieving relevant medical knowledge, and generating responses using a combination of retrieval and generation techniques.

## Features

- RAG-enhanced chatbot for clinical decision support
- Modern, responsive UI
- Detailed analysis of medical queries
- Retrieval of relevant medical knowledge
- Generation of differential diagnoses

## Prerequisites

- Python 3.8+
- Django 4.2+
- Google Gemini API key

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd cdss_chatbot
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the variables in the `.env` file with your own values

```bash
cp .env.example .env
# Edit the .env file with your preferred editor
```

## Running the Application

1. Apply migrations:

```bash
python manage.py migrate
```

2. Start the development server:

```bash
python manage.py runserver
```

3. Open your browser and navigate to:

```
http://127.0.0.1:8000/rag-chatbot/
```

## Usage

1. Type your medical query in the input field
2. Press Enter or click the Send button
3. View the chatbot's response and the detailed analysis

## Project Structure

- `cdss_project/`: Django project settings
- `Rag/`: Django app containing the RAG system and views
  - `rag_system.py`: Implementation of the RAG model
  - `views.py`: Django views for handling requests
  - `urls.py`: URL routing for the application
  - `templates/`: HTML templates for the frontend
- `sample_medical_knowledge.json`: Sample medical knowledge for the RAG system
- `static/`: Static files for the application

## Deployment

### GitHub Deployment

1. Create a GitHub repository
2. Initialize git and push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <github-repo-url>
git push -u origin main
```

### Vercel Deployment

1. Create a Vercel account at [vercel.com](https://vercel.com) if you don't have one
2. Install the Vercel CLI:

```bash
npm install -g vercel
```

3. Deploy to Vercel:

```bash
vercel login
vercel
```

4. Configure the environment variables in the Vercel dashboard:
   - `DJANGO_SECRET_KEY`
   - `GEMINI_API_KEY`
   - `DEBUG` (set to "False" for production)
   - `VERCEL` (set to "1")

5. Your application will be available at the Vercel-assigned URL

## How It Works

1. The user submits a medical query through the frontend
2. The query is sent to the Django backend
3. The RAG system processes the query:
   - Retrieves relevant medical knowledge from the vector database
   - Analyzes the query and retrieved knowledge
   - Generates a response using the Gemini model
4. The response is sent back to the frontend and displayed to the user

## Customization

You can customize the RAG system by:

1. Adding more medical knowledge to `sample_medical_knowledge.json`
2. Modifying the RAG system parameters in `rag_system.py`
3. Updating the frontend design in `templates/chatbot_app/rag_chatbot.html`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 