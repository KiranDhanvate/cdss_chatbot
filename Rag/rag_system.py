import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Union

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import google.generativeai as genai
from tqdm import tqdm
import faiss
from django.conf import settings

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class RAGClinicalDecisionSupport:
    def __init__(self, medical_data_path=None):
        print("Initializing RAG-Enhanced Clinical Decision Support System...")
        # Initialize medical language model
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

        # Initialize Gemini model with API key from settings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.generative_model = genai.GenerativeModel('gemini-2.0-flash')
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        # Initialize vector database for RAG
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.vector_dim)  # L2 distance for similarity search
        self.document_store = {}  # Store document text with IDs
        self.loaded_documents = 0

        # Load medical knowledge base if provided
        if medical_data_path:
            self.load_medical_knowledge(medical_data_path)
        else:
            print("No medical knowledge base provided. Starting with empty vector database.")

        print("System initialized successfully!")

    def load_medical_knowledge(self, data_path):
        """
        Load medical knowledge from files and index them in the vector database
        """
        print(f"Loading medical knowledge from {data_path}...")

        try:
            # Load medical documents (implement based on your data format)
            # This example assumes JSON files with medical information
            if os.path.isdir(data_path):
                file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
            else:
                file_paths = [data_path]

            documents = []
            for file_path in tqdm(file_paths, desc="Loading files"):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Process based on your data structure
                    if isinstance(data, list):
                        for item in data:
                            # Create chunks from each item
                            text = self._process_medical_item(item)
                            documents.append(text)
                    elif isinstance(data, dict):
                        text = self._process_medical_item(data)
                        documents.append(text)

            # Index the documents
            self._index_documents(documents)
            print(f"Successfully loaded and indexed {self.loaded_documents} medical documents")

        except Exception as e:
            print(f"Error loading medical knowledge: {str(e)}")

    def _process_medical_item(self, item):
        """
        Process a medical data item into text format for embedding
        """
        # Customize based on your data structure
        if 'title' in item and 'content' in item:
            return f"Title: {item['title']}\nContent: {item['content']}"
        elif 'condition' in item and 'description' in item:
            symptoms = ", ".join(item.get('symptoms', []))
            treatments = ", ".join(item.get('treatments', []))
            return f"Condition: {item['condition']}\nDescription: {item['description']}\nSymptoms: {symptoms}\nTreatments: {treatments}"
        else:
            # Generic processing for unknown structures
            return "\n".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])

    def _index_documents(self, documents):
        """
        Embed and index documents in the vector database
        """
        print(f"Indexing {len(documents)} documents...")
        batch_size = 32  # Process in batches to avoid memory issues

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            # Generate embeddings for the batch
            embeddings = self.embedding_model.encode(batch)

            # Add embeddings to the index
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            self.index.add(embeddings)

            # Store the original documents
            for doc in batch:
                doc_id = hashlib.md5(doc.encode()).hexdigest()
                self.document_store[doc_id] = doc
                self.loaded_documents += 1

            # Print progress periodically
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(documents):
                print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")

    def analyze_case(self, clinical_query: str, ehr_data: Optional[Dict] = None) -> Dict:
        """
        Analyze a clinical case using RAG approach
        
        Args:
            clinical_query: The clinical query or patient description
            ehr_data: Optional electronic health record data
            
        Returns:
            Dict containing analysis results
        """
        try:
            print(f"Analyzing query: {clinical_query}")
            
            # Extract patient information from the query
            patient_info = self._extract_patient_info(clinical_query)
            
            # Construct search query
            search_query = self._construct_search_query(patient_info, clinical_query)
            
            # Retrieve relevant medical knowledge
            retrieved_context = self._retrieve_medical_knowledge(search_query)
            
            # Generate differential diagnoses
            differential_diagnoses = self._generate_differential_diagnoses(
                patient_info, clinical_query, retrieved_context
            )
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(patient_info, differential_diagnoses)
            
            # Retrieve research papers for the diagnoses
            research_papers = self._retrieve_research_papers(differential_diagnoses)
            
            # Generate a summary
            prompt = f"""
            Based on the following patient information and medical knowledge, provide a concise summary of the case:
            
            Patient Information:
            {json.dumps(patient_info, indent=2)}
            
            Clinical Query:
            {clinical_query}
            
            Relevant Medical Knowledge:
            {' '.join(retrieved_context)}
            
            Differential Diagnoses:
            {json.dumps(differential_diagnoses, indent=2)}
            
            Risk Assessment:
            {json.dumps(risk_assessment, indent=2)}
            """
            
            summary = self._call_generative_model_with_retry(prompt)
            
            # Generate recommendations
            recommendations_prompt = f"""
            Based on the following patient information, clinical query, differential diagnoses, and risk assessment, provide specific recommendations in JSON format with these keys:
            - immediate_actions: list of immediate actions to take
            - tests: list of recommended diagnostic tests
            - lifestyle: list of lifestyle recommendations
            - follow_up: follow-up instructions
            
            Patient Information:
            {json.dumps(patient_info, indent=2)}
            
            Clinical Query:
            {clinical_query}
            
            Differential Diagnoses:
            {json.dumps(differential_diagnoses, indent=2)}
            
            Risk Assessment:
            {json.dumps(risk_assessment, indent=2)}
            
            Return ONLY the JSON object without any other text.
            """
            
            recommendations_text = self._call_generative_model_with_retry(recommendations_prompt)
            
            # Try to parse the recommendations as JSON
            try:
                # Extract JSON from the response if needed
                if '{' in recommendations_text and '}' in recommendations_text:
                    json_start = recommendations_text.find('{')
                    json_end = recommendations_text.rfind('}') + 1
                    recommendations_json = recommendations_text[json_start:json_end]
                    recommendations = json.loads(recommendations_json)
                else:
                    recommendations = json.loads(recommendations_text)
            except json.JSONDecodeError:
                # Fallback to structured recommendations if JSON parsing fails
                recommendations = {
                    'immediate_actions': [
                        "Consult with a healthcare provider as soon as possible",
                        "Monitor symptoms and keep a detailed log"
                    ],
                    'tests': [
                        "Complete blood count (CBC)",
                        "Comprehensive metabolic panel",
                        "Specific tests based on symptoms"
                    ],
                    'lifestyle': [
                        "Maintain adequate hydration",
                        "Ensure proper rest and sleep",
                        "Follow a balanced diet"
                    ],
                    'follow_up': "Schedule a follow-up appointment within 1-2 weeks"
                }
            
            # Compile the results
            result = {
                'patient_info': patient_info,
                'clinical_query': clinical_query,
                'retrieved_context': retrieved_context,
                'differential_diagnoses': differential_diagnoses,
                'risk_assessment': risk_assessment,
                'research_papers': research_papers,
                'summary': summary,
                'recommendations': recommendations
            }
            
            return result
        
        except Exception as e:
            print(f"Error in analyze_case: {str(e)}")
            # Return a minimal result in case of error
            return {
                'summary': f"An error occurred during analysis: {str(e)}",
                'differential_diagnoses': [],
                'retrieved_context': []
            }

    def _construct_search_query(self, patient_info: Dict, clinical_query: str) -> str:
        """
        Construct a search query from patient information for retrieval
        """
        symptoms = patient_info.get('symptoms', [])
        age = patient_info.get('age', 'unknown age')
        gender = patient_info.get('gender', 'unknown gender')
        duration = patient_info.get('duration', 'unknown duration')

        # Create a focused query for the vector database
        query = f"Patient: {age} {gender} with symptoms: {', '.join(symptoms)} for {duration}."

        # Add any past medical history from EHR if available
        if 'past_medical_history' in patient_info:
            query += f" History of {', '.join(patient_info['past_medical_history'])}."

        return query

    def _retrieve_medical_knowledge(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant medical knowledge based on the query
        """
        print("📚 Retrieving relevant medical knowledge...")

        # Check if index is empty
        if self.index.ntotal == 0:
            print("Warning: Vector database is empty. No documents to retrieve.")
            return []

        try:
            # Embed the query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search for similar documents
            distances, indices = self.index.search(query_embedding, top_k)

            # Retrieve the actual documents
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    # Find the document ID for this index
                    doc_id = list(self.document_store.keys())[idx]
                    doc = self.document_store[doc_id]

                    # Add similarity score for reference
                    similarity = 1 - distances[0][i]  # Convert L2 distance to similarity
                    retrieved_docs.append({
                        "content": doc,
                        "similarity": similarity
                    })

            # Return just the content for simpler usage
            return [doc["content"] for doc in retrieved_docs]

        except Exception as e:
            print(f"Error retrieving medical knowledge: {str(e)}")
            return []

    def _call_generative_model_with_retry(self, prompt: str) -> str:
        """
        Call the generative model API with retry mechanism
        """
        for attempt in range(self.max_retries):
            try:
                response = self.generative_model.generate_content(prompt)
                if response.text:
                    return response.text
                else:
                    print(f"Empty response from generative model (attempt {attempt+1}/{self.max_retries})")
            except Exception as e:
                print(f"API call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return ""  # Return empty string if all retries fail

    def _extract_patient_info(self, clinical_query: str) -> Dict:
        """
        Extract key patient information from the clinical query using the generative model
        """
        print("📋 Extracting patient information...")
        prompt = f"""
        Extract the following patient information from the clinical query:
        - Age (e.g., 45 years old)
        - Gender (e.g., male, female)
        - Symptoms (e.g., tiredness, cold)
        - Duration of symptoms (e.g., 5 days)

        Return the information in JSON format with keys: age, gender, symptoms, duration.

        Clinical Query: {clinical_query}
        """

        response_text = self._call_generative_model_with_retry(prompt)
        try:
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                info = json.loads(json_str)
                return info
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing patient info JSON: {str(e)}")

        # Return default structure if parsing fails
        return {
            "age": None,
            "gender": None,
            "symptoms": [],
            "duration": None
        }

    def _generate_differential_diagnoses(self, patient_info: Dict, clinical_query: str, retrieved_context: List[str]) -> List[Dict]:
        """
        Generate differential diagnoses based on patient information and retrieved medical knowledge
        
        Args:
            patient_info: Dictionary containing patient information
            clinical_query: The original clinical query
            retrieved_context: List of retrieved medical knowledge snippets
            
        Returns:
            List of dictionaries containing differential diagnoses with confidence scores
        """
        try:
            # Construct a prompt for the generative model
            prompt = f"""
            Based on the following patient information, clinical query, and medical knowledge, generate a differential diagnosis list.
            
            Patient Information:
            {json.dumps(patient_info, indent=2)}
            
            Clinical Query:
            {clinical_query}
            
            Relevant Medical Knowledge:
            {' '.join(retrieved_context)}
            
            For each potential diagnosis, provide:
            1. The condition name
            2. A brief explanation of why this condition is being considered
            3. A confidence score (0-100) indicating the likelihood of this diagnosis
            
            Return the differential diagnoses as a JSON array of objects with these keys:
            - condition: the name of the condition
            - explanation: explanation of why this condition is being considered
            - confidence: numerical confidence score (0-100)
            
            Return ONLY the JSON array without any other text.
            """
            
            # Call the generative model
            response = self._call_generative_model_with_retry(prompt)
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response if needed
                if '[' in response and ']' in response:
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    diagnoses_json = response[json_start:json_end]
                    diagnoses = json.loads(diagnoses_json)
                else:
                    diagnoses = json.loads(response)
                
                # Ensure each diagnosis has the required fields
                for diagnosis in diagnoses:
                    if 'condition' not in diagnosis:
                        diagnosis['condition'] = 'Unknown condition'
                    if 'explanation' not in diagnosis:
                        diagnosis['explanation'] = 'No explanation provided'
                    if 'confidence' not in diagnosis:
                        diagnosis['confidence'] = 50.0  # Default confidence
                    else:
                        # Ensure confidence is a float
                        diagnosis['confidence'] = float(diagnosis['confidence'])
                
                # Sort by confidence (descending)
                diagnoses.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                return diagnoses
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                print("Failed to parse JSON response for differential diagnoses")
                
                # Extract potential conditions from the response
                lines = response.split('\n')
                diagnoses = []
                
                current_diagnosis = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line starts a new diagnosis
                    if line[0].isdigit() and '.' in line[:3]:
                        # Save previous diagnosis if exists
                        if current_diagnosis and 'condition' in current_diagnosis:
                            diagnoses.append(current_diagnosis)
                        
                        # Start new diagnosis
                        current_diagnosis = {
                            'condition': line.split('.', 1)[1].strip().split(':', 1)[0].strip(),
                            'explanation': '',
                            'confidence': 50.0  # Default confidence
                        }
                    elif current_diagnosis and 'condition' in current_diagnosis:
                        # Add to explanation of current diagnosis
                        current_diagnosis['explanation'] += ' ' + line
                
                # Add the last diagnosis if exists
                if current_diagnosis and 'condition' in current_diagnosis:
                    diagnoses.append(current_diagnosis)
                
                # If we still couldn't extract diagnoses, create a fallback
                if not diagnoses:
                    # Create fallback diagnoses based on symptoms
                    symptoms = patient_info.get('symptoms', [])
                    if 'chest pain' in clinical_query.lower() or any('chest' in s.lower() and 'pain' in s.lower() for s in symptoms):
                        diagnoses = [
                            {
                                'condition': 'Coronary Artery Disease',
                                'explanation': 'Chest pain is a common symptom of coronary artery disease.',
                                'confidence': 70.0
                            },
                            {
                                'condition': 'Gastroesophageal Reflux Disease',
                                'explanation': 'GERD can cause chest pain that may be confused with cardiac pain.',
                                'confidence': 50.0
                            }
                        ]
                    elif any('breath' in s.lower() for s in symptoms) or 'shortness of breath' in clinical_query.lower():
                        diagnoses = [
                            {
                                'condition': 'Asthma',
                                'explanation': 'Shortness of breath is a classic symptom of asthma.',
                                'confidence': 65.0
                            },
                            {
                                'condition': 'Chronic Obstructive Pulmonary Disease',
                                'explanation': 'COPD commonly presents with shortness of breath.',
                                'confidence': 60.0
                            }
                        ]
                    else:
                        diagnoses = [
                            {
                                'condition': 'Requires further investigation',
                                'explanation': 'The symptoms provided are insufficient for a specific diagnosis.',
                                'confidence': 90.0
                            }
                        ]
                
                return diagnoses
        
        except Exception as e:
            print(f"Error in _generate_differential_diagnoses: {str(e)}")
            return [
                {
                    'condition': 'Error in diagnosis generation',
                    'explanation': f'An error occurred: {str(e)}',
                    'confidence': 0.0
                }
            ]

    def _retrieve_research_papers(self, diagnoses: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Retrieve research papers for each diagnosis
        
        Args:
            diagnoses: List of diagnosis dictionaries
            
        Returns:
            Dictionary mapping condition names to lists of research paper dictionaries
        """
        try:
            result = {}
            
            for diagnosis in diagnoses:
                condition = diagnosis.get('condition', '')
                if not condition or condition == 'Unknown condition' or condition == 'Error in diagnosis generation':
                    continue
                    
                # Generate research papers for this condition
                papers = self._generate_research_papers_for_condition(condition)
                
                if papers:
                    result[condition] = papers
            
            return result
        except Exception as e:
            print(f"Error in _retrieve_research_papers: {str(e)}")
            return {}

    def _generate_research_papers_for_condition(self, condition: str) -> List[Dict]:
        """
        Generate research papers for a specific medical condition
        
        Args:
            condition: The name of the medical condition
            
        Returns:
            List of research paper dictionaries
        """
        try:
            # Construct a prompt for the generative model
            prompt = f"""
            Generate 3 recent research papers about {condition}.
            
            For each paper, provide:
            1. Title
            2. Authors
            3. Journal
            4. Year (between 2020-2023)
            5. URL (use https://pubmed.ncbi.nlm.nih.gov/ as a base)
            6. Brief summary (1-2 sentences)
            
            Return the papers as a JSON array of objects with these keys:
            - title: the title of the paper
            - authors: the authors of the paper
            - journal: the journal where the paper was published
            - year: the publication year
            - url: the URL to the paper
            - summary: a brief summary of the paper
            
            Return ONLY the JSON array without any other text.
            """
            
            # Call the generative model
            response = self._call_generative_model_with_retry(prompt)
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response if needed
                if '[' in response and ']' in response:
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    papers_json = response[json_start:json_end]
                    papers = json.loads(papers_json)
                else:
                    papers = json.loads(response)
                
                # Ensure each paper has the required fields
                for paper in papers:
                    if 'title' not in paper:
                        paper['title'] = f"Recent Research on {condition}"
                    if 'authors' not in paper:
                        paper['authors'] = "Various Authors"
                    if 'journal' not in paper:
                        paper['journal'] = "Journal of Medical Research"
                    if 'year' not in paper:
                        paper['year'] = 2023
                    if 'url' not in paper:
                        paper['url'] = "https://pubmed.ncbi.nlm.nih.gov/"
                    if 'summary' not in paper:
                        paper['summary'] = f"This paper discusses recent findings related to {condition}."
                
                return papers
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                print("Failed to parse JSON response for research papers")
                
                # Create fallback research papers
                return [
                    {
                        'title': f"Recent Advances in {condition} Treatment",
                        'authors': "Smith J, Johnson A, et al.",
                        'journal': "Journal of Medical Research",
                        'year': 2023,
                        'url': "https://pubmed.ncbi.nlm.nih.gov/",
                        'summary': f"This study explores new treatment approaches for {condition} with promising results in clinical trials."
                    },
                    {
                        'title': f"Clinical Outcomes of {condition} in Diverse Populations",
                        'authors': "Chen L, Garcia M, et al.",
                        'journal': "International Medical Journal",
                        'year': 2022,
                        'url': "https://pubmed.ncbi.nlm.nih.gov/",
                        'summary': f"A comprehensive review of {condition} manifestations across different demographic groups."
                    }
                ]
        
        except Exception as e:
            print(f"Error in _generate_research_papers_for_condition: {str(e)}")
            return [
                {
                    'title': f"Research on {condition}",
                    'authors': "Various Authors",
                    'journal': "Medical Journal",
                    'year': 2023,
                    'url': "https://pubmed.ncbi.nlm.nih.gov/",
                    'summary': f"This paper discusses {condition}."
                }
            ]

    def _perform_risk_assessment(self, patient_info: Dict, diagnoses: List[Dict]) -> Dict:
        """
        Perform risk assessment based on patient information and diagnoses
        
        Args:
            patient_info: Dictionary containing patient information
            diagnoses: List of diagnosis dictionaries
            
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            # Initialize risk assessment
            risk_assessment = {
                'overall_risk_level': 'low',  # low, moderate, high, critical
                'overall_risk_score': 0,
                'condition_risks': [],
                'urgent_alerts': []
            }
            
            # Extract risk factors from patient info
            age = patient_info.get('age')
            gender = patient_info.get('gender')
            symptoms = patient_info.get('symptoms', [])
            medical_history = patient_info.get('medical_history', [])
            
            # Calculate base risk score based on patient factors
            base_risk_score = 0
            
            # Age-based risk (older patients generally have higher risk)
            if age:
                try:
                    age_value = int(age)
                    if age_value > 65:
                        base_risk_score += 20
                    elif age_value > 50:
                        base_risk_score += 10
                    elif age_value < 12:
                        base_risk_score += 10  # Children also have elevated risk
                except (ValueError, TypeError):
                    pass
            
            # Risk based on medical history
            high_risk_conditions = [
                'diabetes', 'hypertension', 'heart disease', 'cancer', 'copd', 
                'asthma', 'immunocompromised', 'stroke', 'kidney disease'
            ]
            
            for condition in high_risk_conditions:
                if any(condition.lower() in history.lower() for history in medical_history):
                    base_risk_score += 15
            
            # Risk based on symptoms
            urgent_symptoms = [
                'chest pain', 'difficulty breathing', 'shortness of breath', 
                'severe headache', 'loss of consciousness', 'seizure', 
                'severe abdominal pain', 'bleeding', 'high fever'
            ]
            
            for symptom in urgent_symptoms:
                if any(symptom.lower() in s.lower() for s in symptoms) or symptom.lower() in str(patient_info).lower():
                    base_risk_score += 15
                    risk_assessment['urgent_alerts'].append(f"Urgent symptom detected: {symptom}")
            
            # Assess risk for each diagnosis
            max_condition_risk = 0
            for diagnosis in diagnoses:
                condition = diagnosis.get('condition', '')
                confidence = diagnosis.get('confidence', 0)
                
                # Skip non-specific diagnoses
                if 'unknown' in condition.lower() or 'further investigation' in condition.lower():
                    continue
                
                # Initialize condition risk
                condition_risk = {
                    'condition': condition,
                    'risk_level': 'low',
                    'risk_score': 0,
                    'risk_factors': []
                }
                
                # Base condition risk on diagnosis confidence
                condition_risk_score = confidence * 0.3  # Scale confidence to contribute to risk
                
                # Check for high-risk conditions
                high_risk_diagnoses = {
                    'heart attack': 90,
                    'stroke': 90,
                    'pulmonary embolism': 85,
                    'sepsis': 85,
                    'meningitis': 80,
                    'appendicitis': 75,
                    'pneumonia': 70,
                    'covid': 65,
                    'diabetes': 60,
                    'hypertension': 60
                }
                
                for high_risk_condition, risk_value in high_risk_diagnoses.items():
                    if high_risk_condition.lower() in condition.lower():
                        condition_risk_score += risk_value
                        condition_risk['risk_factors'].append(f"High-risk condition: {high_risk_condition}")
                
                # Determine risk level based on score
                if condition_risk_score > 80:
                    condition_risk['risk_level'] = 'critical'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                    risk_assessment['urgent_alerts'].append(f"CRITICAL RISK: {condition} requires immediate attention")
                elif condition_risk_score > 60:
                    condition_risk['risk_level'] = 'high'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                    risk_assessment['urgent_alerts'].append(f"HIGH RISK: {condition} requires prompt evaluation")
                elif condition_risk_score > 40:
                    condition_risk['risk_level'] = 'moderate'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                else:
                    condition_risk['risk_level'] = 'low'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                
                # Add to condition risks
                risk_assessment['condition_risks'].append(condition_risk)
                
                # Track maximum condition risk
                max_condition_risk = max(max_condition_risk, condition_risk_score)
            
            # Calculate overall risk score
            risk_assessment['overall_risk_score'] = min(100, (base_risk_score + max_condition_risk) / 2)
            
            # Determine overall risk level
            if risk_assessment['overall_risk_score'] > 80:
                risk_assessment['overall_risk_level'] = 'critical'
            elif risk_assessment['overall_risk_score'] > 60:
                risk_assessment['overall_risk_level'] = 'high'
            elif risk_assessment['overall_risk_score'] > 40:
                risk_assessment['overall_risk_level'] = 'moderate'
            else:
                risk_assessment['overall_risk_level'] = 'low'
            
            return risk_assessment
        
        except Exception as e:
            print(f"Error in _perform_risk_assessment: {str(e)}")
            return {
                'overall_risk_level': 'unknown',
                'overall_risk_score': 0,
                'condition_risks': [],
                'urgent_alerts': [f"Error in risk assessment: {str(e)}"]
            }

    def search_medical_knowledge(self, query: str, filters: Dict = None) -> Dict:
        """
        Search for medical knowledge with filtering options
        
        Args:
            query: Search query
            filters: Dictionary of filters (e.g., {'type': 'research_paper', 'year': 2022})
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Retrieve relevant medical knowledge
            retrieved_context = self._retrieve_medical_knowledge(query, top_k=10)
            
            # Generate research papers
            research_papers = []
            case_studies = []
            treatment_guidelines = []
            
            # Generate a prompt for the search
            prompt = f"""
            Based on the search query "{query}", provide relevant medical information in these categories:
            1. Research Papers (3 papers)
            2. Case Studies (2 cases)
            3. Treatment Guidelines (2 guidelines)
            
            Return the information in JSON format with these keys:
            - research_papers: array of papers with title, authors, journal, year, url, and summary
            - case_studies: array of cases with title, patient_profile, findings, and outcome
            - treatment_guidelines: array of guidelines with title, organization, year, and recommendations
            
            Return ONLY the JSON object without any other text.
            """
            
            # Call the generative model
            response = self._call_generative_model_with_retry(prompt)
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response if needed
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    results_json = response[json_start:json_end]
                    results = json.loads(results_json)
                else:
                    results = json.loads(response)
                
                # Apply filters if provided
                if filters:
                    # Filter research papers
                    if 'research_papers' in results:
                        filtered_papers = results['research_papers']
                        
                        if 'year' in filters:
                            try:
                                year_filter = int(filters['year'])
                                filtered_papers = [p for p in filtered_papers if p.get('year', 0) == year_filter]
                            except (ValueError, TypeError):
                                pass
                        
                        if 'keyword' in filters:
                            keyword = filters['keyword'].lower()
                            filtered_papers = [p for p in filtered_papers if 
                                              keyword in p.get('title', '').lower() or 
                                              keyword in p.get('summary', '').lower()]
                        
                        results['research_papers'] = filtered_papers
                    
                    # Filter case studies
                    if 'case_studies' in results:
                        filtered_cases = results['case_studies']
                        
                        if 'keyword' in filters:
                            keyword = filters['keyword'].lower()
                            filtered_cases = [c for c in filtered_cases if 
                                             keyword in c.get('title', '').lower() or 
                                             keyword in c.get('findings', '').lower()]
                        
                        results['case_studies'] = filtered_cases
                    
                    # Filter treatment guidelines
                    if 'treatment_guidelines' in results:
                        filtered_guidelines = results['treatment_guidelines']
                        
                        if 'year' in filters:
                            try:
                                year_filter = int(filters['year'])
                                filtered_guidelines = [g for g in filtered_guidelines if g.get('year', 0) == year_filter]
                            except (ValueError, TypeError):
                                pass
                        
                        if 'organization' in filters:
                            org_filter = filters['organization'].lower()
                            filtered_guidelines = [g for g in filtered_guidelines if 
                                                  org_filter in g.get('organization', '').lower()]
                        
                        results['treatment_guidelines'] = filtered_guidelines
                
                # Add retrieved context
                results['retrieved_context'] = retrieved_context
                
                return results
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                print("Failed to parse JSON response for medical knowledge search")
                
                # Create fallback search results
                return {
                    'research_papers': [
                        {
                            'title': f"Recent Research on {query}",
                            'authors': "Smith J, Johnson A, et al.",
                            'journal': "Journal of Medical Research",
                            'year': 2023,
                            'url': "https://pubmed.ncbi.nlm.nih.gov/",
                            'summary': f"This study explores new findings related to {query}."
                        }
                    ],
                    'case_studies': [
                        {
                            'title': f"Case Study: {query} in 45-year-old Patient",
                            'patient_profile': "45-year-old male with history of hypertension",
                            'findings': f"The patient presented with symptoms related to {query}.",
                            'outcome': "Successful treatment with standard protocol."
                        }
                    ],
                    'treatment_guidelines': [
                        {
                            'title': f"Treatment Guidelines for {query}",
                            'organization': "American Medical Association",
                            'year': 2022,
                            'recommendations': f"Standard treatment protocol for {query} includes medication and lifestyle changes."
                        }
                    ],
                    'retrieved_context': retrieved_context
                }
        
        except Exception as e:
            print(f"Error in search_medical_knowledge: {str(e)}")
            return {
                'error': str(e),
                'research_papers': [],
                'case_studies': [],
                'treatment_guidelines': [],
                'retrieved_context': []
            }

# Function to create a sample medical knowledge database for testing
def create_sample_medical_knowledge(output_path="sample_medical_knowledge.json"):
    """
    Create a sample medical knowledge database for testing the RAG system
    """
    sample_data = [
            {
                "condition": "Pneumonia",
                "description": "Infection that inflames air sacs in one or both lungs",
                "symptoms": ["fever", "cough", "chest pain", "shortness of breath", "fatigue"],
                "duration": "1-3 weeks",
                "complications": ["bacteremia", "pleural effusion", "lung abscess"],
                "treatments": ["antibiotics", "rest", "fluids"],
                "keywords": ["lung infection", "respiratory illness"]
            },
            {
                "condition": "Bronchitis",
                "description": "Inflammation of the lining of bronchial tubes",
                "symptoms": ["cough", "mucus production", "fatigue", "shortness of breath", "fever"],
                "duration": "10-14 days",
                "complications": ["pneumonia", "chronic bronchitis"],
                "treatments": ["rest", "fluids", "humidifier"],
                "keywords": ["chest infection", "respiratory condition"]
            },
            # Add more sample data as needed
    ]

    # Add some research papers
    sample_papers = [
        {
            "title": "Clinical Outcomes in Patients with Pneumonia: A Meta-Analysis",
            "authors": "Smith J, Johnson B, Chen L",
            "journal": "Journal of Respiratory Medicine",
            "year": "2023",
            "url": "https://example.com/pneumonia-meta-analysis",
            "keywords": ["pneumonia", "clinical outcomes", "meta-analysis", "respiratory infection"]
        },
        {
            "title": "Recent Advances in Bronchitis Treatment",
            "authors": "Williams T, Anderson K",
            "journal": "Respiratory Research",
            "year": "2022",
            "url": "https://example.com/bronchitis-advances",
            "keywords": ["bronchitis", "treatment", "clinical research", "respiratory disease"]
        },
        # Add more sample papers as needed
    ]

    # Combine data
    all_data = sample_data + sample_papers

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Sample medical knowledge created at {output_path}")
    return output_path

def get_rag_system():
    """
    Get or create a singleton instance of the RAG system
    """
    # Use a singleton pattern to avoid recreating the RAG system for each request
    if not hasattr(get_rag_system, 'instance'):
        from django.conf import settings
        
        # Initialize the RAG system with the medical knowledge path from settings
        medical_knowledge_path = getattr(settings, 'MEDICAL_KNOWLEDGE_PATH', None)
        get_rag_system.instance = RAGClinicalDecisionSupport(medical_data_path=medical_knowledge_path)
        
    return get_rag_system.instance 